from pathlib import Path
import numpy as np
import sympy
import pytorch_kinematics as pk

SO101_JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
)

URDF_PATH = Path(__file__).resolve().parent.parent / "assets" / "so101" / "robot.urdf"

# Link lengths from the simplified diagram model (approximate values in meters)
# Based on the URDF and diagram:
# L1: base to shoulder lift axis (height offset) - not used directly in planar IK
# L2: shoulder lift to elbow flex (upper arm length)
# L3: elbow flex to wrist flex (lower arm length)
# L4: wrist flex to gripper frame (wrist + gripper length)

# From analyzing the URDF transforms:
# shoulder_lift joint is at z=0.0624 from base
# upper_arm_link to lower_arm_link: elbow joint
# The key lengths for the planar 3-link arm (shoulder_lift, elbow_flex, wrist_flex):
L2 = 0.11257  # upper arm length (shoulder_lift to elbow_flex)
L3 = 0.1349   # lower arm length (elbow_flex to wrist_flex)
L4 = 0.0981274  # wrist to gripper_frame (from gripper_frame_joint xyz)

# Offset from base to shoulder pan axis
BASE_HEIGHT = 0.0624  # z-offset to shoulder_pan


def so101_downturned_ik_symbolic(
    x: sympy.Symbol,
    y: sympy.Symbol,
    z: sympy.Symbol,
    yaw: sympy.Symbol,
) -> dict[str, sympy.Expr]:
    """
    Return a dict mapping each joint name to a sympy expression in (x, y, z, yaw).
    
    Parameters
    ----------
    x, y, z, yaw : sympy.Symbol
        Symbols for end-effector position and yaw.
    
    Returns
    -------
    dict
        Mapping from each key in SO101_JOINT_NAMES to a sympy expression (joint angle in radians).
        Should be None if no solution within joint limits is found.
    """
    # shoulder_pan: rotation about base Z axis to point toward (x, y)
    shoulder_pan = sympy.atan2(y, x)
    
    # For the remaining joints, we work in the plane defined by shoulder_pan
    # The radial distance from base Z axis
    r = sympy.sqrt(x**2 + y**2)
    
    # The effective z height relative to shoulder_lift joint
    # shoulder_lift joint is at height BASE_HEIGHT from base
    z_eff = z - BASE_HEIGHT
    
    # For downturned gripper (z-axis pointing down), the wrist needs to position
    # the gripper such that after the wrist_flex and wrist_roll, the gripper points down
    #
    # In the simplified model:
    # - The gripper_frame is at the target (x, y, z)
    # - The wrist_flex joint is L4 away from gripper_frame along the gripper's approach direction
    # - For downturned gripper, the approach is along -Z, so wrist is at (x, y, z + L4)
    
    z_wrist = z_eff + L4
    
    # Now we have a 2-link planar IK problem for shoulder_lift and elbow_flex
    # Target for wrist_flex joint: (r, z_wrist) in the shoulder plane
    # Link lengths: L2 (shoulder_lift to elbow), L3 (elbow to wrist_flex)
    
    # Distance from shoulder_lift to wrist target
    d = sympy.sqrt(r**2 + z_wrist**2)
    
    # Law of cosines for elbow_flex angle
    # d^2 = L2^2 + L3^2 - 2*L2*L3*cos(pi - elbow_flex)
    # cos(elbow_flex) = (L2^2 + L3^2 - d^2) / (2*L2*L3)
    cos_elbow = (L2**2 + L3**2 - d**2) / (2 * L2 * L3)
    
    # elbow_flex: using atan2 for proper quadrant handling
    # We use the "elbow up" configuration (negative elbow angle in our convention)
    sin_elbow_pos = sympy.sqrt(1 - cos_elbow**2)
    elbow_flex = sympy.atan2(-sin_elbow_pos, cos_elbow)  # elbow up config
    
    # shoulder_lift angle
    # Angle to wrist target from horizontal
    alpha = sympy.atan2(z_wrist, r)
    # Angle between L2 and line to wrist (law of cosines)
    # cos(beta) = (L2^2 + d^2 - L3^2) / (2 * L2 * d)
    cos_beta = (L2**2 + d**2 - L3**2) / (2 * L2 * d)
    sin_beta_pos = sympy.sqrt(1 - cos_beta**2)
    beta = sympy.atan2(sin_beta_pos, cos_beta)
    
    # shoulder_lift = alpha + beta (for elbow up config, we subtract)
    shoulder_lift = alpha - beta
    
    # wrist_flex: adjusts the gripper orientation
    # For downturned gripper with given yaw, we need to compute the wrist angles
    # The total rotation from shoulder_lift to gripper must result in gripper pointing down
    #
    # In the plane: shoulder_lift + elbow_flex + wrist_flex should orient the gripper
    # For downturned (pointing -Z in world), the gripper frame has specific orientation
    #
    # Simplified: wrist_flex compensates for shoulder_lift + elbow_flex to achieve downturned
    # wrist_flex = -shoulder_lift - elbow_flex (for straight down in the plane)
    # But we also need to account for the yaw rotation
    
    # For the simplified model with downturned gripper:
    # wrist_flex keeps the gripper pointing down regardless of arm configuration
    # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: добавляем -pi/2 для ориентации захвата вниз
    wrist_flex = -sympy.pi/2 - shoulder_lift - elbow_flex
    
    # wrist_roll: provides the yaw rotation about the gripper's approach axis
    # Since gripper is downturned, wrist_roll directly controls yaw
    wrist_roll = yaw
    
    return {
        "shoulder_pan": shoulder_pan,
        "shoulder_lift": shoulder_lift,
        "elbow_flex": elbow_flex,
        "wrist_flex": wrist_flex,
        "wrist_roll": wrist_roll,
    }


def analytical_ik_so101_downturned(
    x: float, y: float, z: float, yaw: float
) -> dict[str, float]:
    """
    Evaluate the analytical IK formulas numerically and check joint limits.
    
    Parameters
    ----------
    x, y, z: float
        Desired end-effector position (x, y, z) in base frame.
    yaw : float
        Desired yaw angle (radians) in the downturned end-effector plane.
    
    Returns
    -------
    dict
        Mapping from each key in SO101_JOINT_NAMES to a float (joint angle in radians).
        Should be None if no solution within joint limits is found.
    """
    x_sym, y_sym, z_sym, yaw_sym = sympy.symbols("x y z yaw", real=True)
    formulas = so101_downturned_ik_symbolic(x_sym, y_sym, z_sym, yaw_sym)
    func = sympy.lambdify(
        (x_sym, y_sym, z_sym, yaw_sym),
        [formulas[k] for k in SO101_JOINT_NAMES],
        "numpy",
    )
    q = dict(zip(SO101_JOINT_NAMES, func(x, y, z, yaw)))
    
    chain = pk.build_chain_from_urdf(open(URDF_PATH, mode="rb").read())
    serial_chain = pk.SerialChain(chain, "gripper_frame_link", "base_link")
    low, high = serial_chain.get_joint_limits()
    low, high = np.asarray(low), np.asarray(high)
    
    # ИСПРАВЛЕНИЕ: joint уже строка, не нужно .name
    for idx, joint_name in enumerate(SO101_JOINT_NAMES):
        joint_angle = q[joint_name]
        if joint_angle < low[idx] or joint_angle > high[idx]:
            return None
    
    return q


def numerical_ik_so101_downturned(
    x: float, y: float, z: float, yaw: float
) -> dict[str, float] | None:
    """
    Numerical IK for a downturned SO101 pose.
    
    Parameters
    ----------
    x, y, z: float
        Desired end-effector position (x, y, z) in base frame.
    yaw : float
        Desired yaw (radians) in the downturned end-effector plane.
    
    Returns
    -------
    dict
        Mapping from each key in SO101_JOINT_NAMES to a float (joint angle in radians).
        Should be None if no solution within joint limits is found.
    """
    import torch
    from pytorch_kinematics.transforms import transform3d
    
    # Build the target 4x4 transform matrix for the downturned gripper pose
    # For downturned gripper: z-axis points down (-Z world), yaw rotates about Z
    c, s = np.cos(yaw), np.sin(yaw)
    
    # Combined rotation: R = R_z(yaw) * R_x(pi) for downturned gripper
    R = np.array([
        [c, s, 0.0],
        [s, -c, 0.0],
        [0.0, 0.0, -1.0]
    ], dtype=np.float64)
    
    # Build 4x4 transform
    T_target = np.eye(4, dtype=np.float64)
    T_target[:3, :3] = R
    T_target[:3, 3] = [x, y, z]
    
    # Build kinematics chain
    chain = pk.build_chain_from_urdf(open(URDF_PATH, mode="rb").read())
    serial_chain = pk.SerialChain(chain, "gripper_frame_link", "base_link")
    
    # Get joint limits
    low, high = serial_chain.get_joint_limits()
    low, high = np.asarray(low, dtype=np.float64), np.asarray(high, dtype=np.float64)
    
    # Use PseudoInverseIK solver with num_retries and joint_limits (use float32)
    low_f32 = low.astype(np.float32)
    high_f32 = high.astype(np.float32)
    
    ik_solver = pk.PseudoInverseIK(
        serial_chain, 
        num_retries=5,
        joint_limits=torch.tensor(np.column_stack([low_f32, high_f32]), dtype=torch.float32)
    )
    
    # Convert target to Transform3d (use float32)
    T_target_f32 = T_target.astype(np.float32)
    T_target_torch = torch.tensor(T_target_f32, dtype=torch.float32).unsqueeze(0)
    target_transform = transform3d.Transform3d(matrix=T_target_torch)
    
    # Solve IK
    result = ik_solver.solve(target_transform)
    
    # Extract best solution (lowest error)
    # result.solutions is (num_retries, num_problems, dof)
    solutions = result.solutions  # shape: (num_retries, 1, dof)
    
    # Compute total error for each retry
    err_pos = result.err_pos.detach().numpy()  # (num_retries, 1)
    err_rot = result.err_rot.detach().numpy()  # (num_retries, 1)
    total_errors = err_pos + err_rot
    
    # Find the best retry (minimum error)
    best_idx = np.argmin(total_errors[:, 0])
    q_solution = solutions[best_idx, 0].detach().numpy()
    
    # Check if solution converged
    if total_errors[best_idx, 0] > 1e-2:
        return None
    
    # Check joint limits
    for i, angle in enumerate(q_solution):
        if angle < low[i] - 1e-6 or angle > high[i] + 1e-6:
            return None
    
    # Return as dict
    return {name: float(q_solution[i]) for i, name in enumerate(SO101_JOINT_NAMES)}