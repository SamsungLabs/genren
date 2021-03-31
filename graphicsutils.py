import torch, numpy as np, math
import numpy.random as npr
import torch.nn as nn, torch.nn.functional as F
import torch.autograd as autograd

def spherical_d(azimuths, elevations, distances):
    """
    Note: input azimuth and elevation are in degrees
    """
    thetas   = azimuths   * math.pi / 180.0
    phis     = elevations * math.pi / 180.0
    return spherical_rads(thetas, phis, distances)
    
def spherical_rads(azimuths, elevations, distances):
    """
    Each input a vector of length B [in radians].
    Output: B x 3
    """
    camY    = distances  * torch.sin(elevations) 
    temp    = distances  * torch.cos(elevations)
    camX    = temp       * torch.cos(azimuths)
    camZ    = temp       * torch.sin(azimuths)
    return torch.stack([camX, camY, camZ], dim = 1)

def unit_spherical_rads(azimuths, elevations):
    camY    = torch.sin(elevations) 
    temp    = torch.cos(elevations)
    camX    = temp * torch.cos(azimuths)
    camZ    = temp * torch.sin(azimuths)
    return torch.stack([camX, camY, camZ], dim = 1)

def spherical_rads_np(azimuths, elevations, distances):
    """
    Each input a float or np array (length B)
    Output: 3-vector or Bx3 array
    """
    camY    = distances  * np.sin(elevations) 
    temp    = distances  * np.cos(elevations)
    camX    = temp       * np.cos(azimuths)
    camZ    = temp       * np.sin(azimuths)
    return np.array([camX, camY, camZ])

def compute_camera_params(azimuth, elevation, distance, eps=1e-6):
    device  = azimuth.device
    theta   = azimuth   * math.pi / 180.0
    phi     = elevation * math.pi / 180.0
    camY    = distance  * torch.sin(phi)
    temp    = distance  * torch.cos(phi)
    camX    = temp      * torch.cos(theta)
    camZ    = temp      * torch.sin(theta)
    cam_pos = torch.cat([camX, camY, camZ], dim = 0)
    axisZ   = cam_pos #.clone()
    axisY   = torch.FloatTensor([0.0, 1.0, 0.0]).to(device)
    axisX   = torch.cross(axisY, axisZ) + eps
    axisY   = torch.cross(axisZ, axisX)
    cam_mat = torch.stack([axisX, axisY, axisZ])
    l2      = torch.norm(cam_mat, p=2, dim=1).unsqueeze(1)
    cam_mat = cam_mat / l2
    return cam_mat, cam_pos

def compute_camera_params_np(azimuth: float, elevation: float, distance: float):

    theta = np.deg2rad(azimuth)
    phi = np.deg2rad(elevation)

    camY = distance * np.sin(phi)
    temp = distance * np.cos(phi)
    camX = temp * np.cos(theta)
    camZ = temp * np.sin(theta)
    cam_pos = np.array([camX, camY, camZ])

    axisZ = cam_pos.copy()
    axisY = np.array([0, 1, 0])
    axisX = np.cross(axisY, axisZ)
    axisY = np.cross(axisZ, axisX)

    cam_mat = np.array([axisX, axisY, axisZ])
    l2 = np.atleast_1d(np.linalg.norm(cam_mat, 2, 1))
    l2[l2 == 0] = 1
    cam_mat = cam_mat / np.expand_dims(l2, 1)

    return torch.FloatTensor(cam_mat), torch.FloatTensor(cam_pos)

#############################################################################

### Rotation Handling ###

def normalize_vector(v, return_mag = False):
    batch = v.shape[0]
    v_mag = torch.clamp( torch.sqrt(v.pow(2).sum(1)), min=1e-7)
    #v_mag = torch.max(v_mag, torch.autograd.Variable(torch.FloatTensor([1e-8]).cuda()))
    v_mag = v_mag.view(batch,1).expand(batch,v.shape[1])
    v = v / v_mag
    if return_mag:
        return v, v_mag[:,0]
    else:
        return v

def compute_rotation_matrix_from_ortho6d(ortho6d):
    x_raw = ortho6d[:,0:3]#batch*3
    y_raw = ortho6d[:,3:6]#batch*3
        
    x = normalize_vector(x_raw) #batch*3
    z = torch.cross(x,y_raw) #batch*3
    z = normalize_vector(z) #batch*3
    y = torch.cross(z,x) #batch*3
        
    x = x.view(-1,3,1)
    y = y.view(-1,3,1)
    z = z.view(-1,3,1)
    return torch.cat((x,y,z), 2) #batch*3*3

class SixDimRotationDecoder(nn.Module):
    def __init__(self):
        super(SixDimRotationDecoder, self).__init__()

    def forward(self, r):
        """ B x 6 --> B x 3 x 3 """
        return compute_rotation_matrix_from_ortho6d(r)

def compute_rotation_matrix_from_intrinsic_Euler_angles_np(phi,theta,psi):
    """
    https://en.wikipedia.org/wiki/Rotation_formalisms_in_three_dimensions#Conversion_formulae_between_formalisms
    """
    sphi   = np.sin(phi)
    stheta = np.sin(theta)
    spsi   = np.sin(psi)
    cphi   = np.cos(phi)
    ctheta = np.cos(theta)
    cpsi   = np.cos(psi)
    return np.array([
            [ ctheta*cpsi, -cphi*spsi + sphi*stheta*cpsi,  sphi*spsi + cphi*stheta*cpsi ],
            [ ctheta*spsi,  cphi*cpsi + sphi*stheta*spsi, -sphi*cpsi + cphi*stheta*spsi ],
            [ -stheta,                       sphi*ctheta,                   cphi*ctheta ] ])

class MinAngleComposedRotationLoss(nn.Module):
    """ Error/loss between rotation matrices """
    def __init__(self):
        super(MinAngleComposedRotationLoss, self).__init__()
        self.eps = 1e-6
        self.mineps = -1.0 + self.eps 
        self.maxeps =  1.0 - self.eps

    def forward(self, R1, R2, mean_out=True):
        """ Inputs: two batches of rotation matrices (B x 3 x 3) """
        if mean_out:
            return _min_angle_of_composed_rotation(R1, R2, self.mineps, self.maxeps).mean()
        else:
            return _min_angle_of_composed_rotation(R1, R2, self.mineps, self.maxeps)

class WeightedMultiHypMinAngComposedRotLoss(nn.Module):
    def __init__(self):
        super(WeightedMultiHypMinAngComposedRotLoss, self).__init__()
        self.rot_dist = MinAngleComposedRotationLoss()

    def forward(self, R_true, R_hyp, pose_probs):
        B, NH, _, _ = R_hyp.shape
        rot_recon_loss = (self.rot_dist(
                            R_true.expand(-1,NH,-1,-1).reshape(B*NH,3,3),
                            R_hyp.view(B*NH,3,3),
                            mean_out=False).view(B,NH) * pose_probs
                          ).sum(dim=1).mean(dim=0)
        return rot_recon_loss

def _min_angle_of_composed_rotation(R1, R2, mineps, maxeps):
    """
    See Huynh, "Metrics for 3D Rotations: Comparison and Analysis", especially ~(25)-(26).
    The exponential map representation of R_diff = R_1 R_2^T = exp([theta u_hat]_X) is a
        rotation matrix representing how to get from R_1 and R_2.
    If R_1 ~ R_2, then R_diff ~ I (i.e., theta ~ 0).
    It can be shown (see the paper, appendix A) that tr(R) = 1 + 2cos(theta) for any R in SO(3).
    So, tr(R_diff) = 1 + 2cos(theta_diff) --> theta_diff = arccos( (tr(R_diff) - 1)/2 )
        where theta_diff measures the magnitude (angle) of the rotation that occurs when applying
        inverse R2, followed by applying R1.

    Inputs: R{1,2} in B x 3 x 3
    Outputs: L in B
    """
    R_diff = torch.bmm(R1, R2.transpose(1,2))
    return torch.acos( 
              torch.clamp(
                  (R_diff[:,0,0] + R_diff[:,1,1] + R_diff[:,2,2] - 1.0) / 2.0, 
                  min = mineps, 
                  max = maxeps
              ) # In [-1, 1]
           ) # In [0, pi]

class RotationNegEntropyLossRonly(nn.Module):
    """
    Computes the negative sum of pairwise distances between rotation hypotheses.
        L = - (1/N_H^2) sum_i sum_j d(R_i, R_j)
    Minimizing this means maximizing the pairwise distance between rotation matrices.
    Output loss is in [0, 1].
    """
    def __init__(self):
        super(RotationNegEntropyLossRonly, self).__init__()
        self.rot_dist = MinAngleComposedRotationLoss()

    def forward(self, R):
        B, NH, _, _ = R.shape
        R1 = R.unsqueeze(1).expand(-1,NH,-1,-1,-1).reshape(B*NH*NH,3,3)
        R2 = R.unsqueeze(2).expand(-1,-1,NH,-1,-1).reshape(B*NH*NH,3,3)
        return -1.0 * self.rot_dist(R1, R2) / math.pi

class RotationNegEntropyLoss(nn.Module):
    """
    Computes the probability weighted negative sum of pairwise distances between rotation hypotheses.
        L = - (1/pi) sum_i sum_j d(R_i, R_j)
    Minimizing this means maximizing the pairwise distance between rotation matrices.
    Output loss is in [0, 1], then negativized.
    """
    def __init__(self):
        super(RotationNegEntropyLoss, self).__init__()
        self.rot_dist = MinAngleComposedRotationLoss()

    def forward(self, R, p):
        B, NH, _, _ = R.shape
        R1 = R.unsqueeze(1).expand(-1,NH,-1,-1,-1).reshape(B*NH*NH,3,3)
        R2 = R.unsqueeze(2).expand(-1,-1,NH,-1,-1).reshape(B*NH*NH,3,3)
        # Distance matrix between rotation matrices
        D = self.rot_dist(R1, R2, mean_out=False).reshape(B,NH,NH) # B * NH * NH, in [0,pi]
        # Outer product of pose probabilities
        p_outer_p = torch.bmm(p.unsqueeze(-1), p.unsqueeze(1))
        # Compute prob weighted pairwise distances:
        #      penalty = sum_{i,j} p_i p_j D(R_i, R_j),
        # and then averaged over the batch.
        pw_d = (p_outer_p * D).sum(-1).sum(-1).mean()
        return -1.0 * pw_d / math.pi

#############################

class QuatRotationDecoder(nn.Module):
    def __init__(self):
        super(QuatRotationDecoder, self).__init__()
    def forward(self, q):
        return quat2mat(q)


def quat2mat(quat):
    """
    Convert quaternion coefficients to rotation matrix.

    Args:
        quat: quaternion as a 4-tuple -- size = [B, 4]
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]

    From:
       https://github.com/ClementPinard/SfmLearner-Pytorch
    Specifically:
       https://github.com/ClementPinard/SfmLearner-Pytorch/blob/master/inverse_warp.py
    MIT Licensed
    Minor modifications by TTAA
    """
    # norm_quat = torch.cat([quat[:,:1].detach()*0 + 1, quat], dim=1)
    # norm_quat = norm_quat/norm_quat.norm(p=2, dim=1, keepdim=True)
    norm_quat = F.normalize(quat, p=2, dim=1)
    w, x, y, z = norm_quat[:,0], norm_quat[:,1], norm_quat[:,2], norm_quat[:,3]
    B = quat.size(0)
    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z
    rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                          2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                          2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2],
                    dim=1).view(B, 3, 3)
    return rotMat

#############################

class AngleTripletDecoder(nn.Module):
    def __init__(self, alim):
        super(AngleTripletDecoder, self).__init__()
        self.angle_limit = alim 

    def forward(self, r):
        """
        Input r (B x 3) is supposed to be a triplet of 3 unconstrained values.
        It is first mapped to an angle (limited by the `angle_limit` field) and
            then each angle is mapped to the 3 elemental rotation matrices (per axis).
        Their product forms the final rotation matrix.

        This can be viewed as generating the yaw, pitch, and roll angles (improper
            Euler angles or Tait–Bryan angles)

        See:
            https://en.wikipedia.org/wiki/Rotation_matrix#In_three_dimensions
        """
        r = self.angle_limit * torch.tanh(r) # [a,b] -> [-1, 1] -> [-AL, AL]
        return self.constrained_forward(r)

    def constrained_forward(self, r):
        alpha = r[:, 0] # phi (roll)
        beta  = r[:, 1] # theta (pitch)
        gamma = r[:, 2] # psi (yaw)
        sin_alpha = torch.sin(alpha)
        sin_beta  = torch.sin(beta)
        sin_gamma = torch.sin(gamma)
        cos_alpha = torch.cos(alpha)
        cos_beta  = torch.cos(beta)
        cos_gamma = torch.cos(gamma)
        R01 = cos_alpha*sin_beta*sin_gamma - sin_alpha*cos_gamma
        R02 = cos_alpha*sin_beta*cos_gamma + sin_alpha*sin_gamma
        R11 = sin_alpha*sin_beta*sin_gamma + cos_alpha*cos_gamma
        R12 = sin_alpha*sin_beta*cos_gamma - cos_alpha*sin_gamma
        R = torch.stack([
                torch.stack([ cos_alpha*cos_beta, R01,                R02 ]),
                torch.stack([ sin_alpha*cos_beta, R11,                R12 ]),
                torch.stack([ -sin_beta,          cos_beta*sin_gamma, cos_beta*cos_gamma ])
        ]).permute(2,0,1)
        return R

def quat_to_tait_bryan(q):
    """
    q   : B x 4 [q_real, qimg1, qimg2, qimg3]
    out : B x 3 [roll, pitch, yaw]
    """
    q = F.normalize(q, p=2, dim=1)
    q_real = q[:, 0]
    q_i    = q[:, 1]
    q_j    = q[:, 2]
    q_k    = q[:, 3]
    roll   = torch.atan2( 2.0*(q_real*q_i + q_j*q_k), 1.0 - 2.0*(q_i**2 + q_j**2) ) # phi = alpha
    yaw    = torch.atan2( 2.0*(q_real*q_k + q_i*q_j), 1.0 - 2.0*(q_j**2 + q_k**2) ) # psi = gamma
    pitch  = torch.asin(  2.0*(q_real*q_j - q_k*q_i) ) # theta = beta
    return torch.cat( (roll.unsqueeze(1), pitch.unsqueeze(1), yaw.unsqueeze(1)), dim=1)

def axis_angle_to_quat(axis, angle):
    """
    axis: B x 3 
    angle: B
    output: q [B x 4] (real, imag)
    """
    t_over_2 = angle / 2.0
    s   = torch.sin(t_over_2)
    q_i = axis[:,0] * s
    q_j = axis[:,1] * s
    q_k = axis[:,2] * s
    q_r = torch.cos(t_over_2)
    return torch.stack( (q_r, q_i, q_j, q_k), dim = 1)

def quat_rot_from_v1_to_v2(v1, v2):
    """
    Gives a quat that rotates v1 to v2.

    v1 : B x 3
    v2 : B x 3
    Output q: B x 4 [q_real, qimg1, qimg2, qimg3]
    """
    B = v1.shape[0]
    # Normalize inputs
    v1 = F.normalize(v1, p=2, dim=1)
    v2 = F.normalize(v2, p=2, dim=1)
    # Compute angles
    dot_prod = torch.bmm( v1.unsqueeze(1), v2.unsqueeze(-1) ).squeeze(-1).squeeze(-1)
    # Mask for telling us which pairs are either parallel or anti-parallel
    mask_parallel      = (dot_prod > 0.9999)
    mask_anti_parallel = (dot_prod < -0.9999)
    # The identity quaternion (which generates R = I)
    identity_quats    = torch.zeros(4)
    identity_quats[0] = 1.0
    # The flip quaternion 
    # A 180 deg rotation can happen about any axis -> q_flip = (0, axis)
    flip_quat    = torch.zeros(4)
    flip_quat[1] = 1.0
    # Standard case
    axis      = torch.cross(v1, v2) # B x 3
    real_part = 1.0 + dot_prod # B
    # Output quaternion container
    quat          = torch.zeros(B, 4)
    quat[:, 0]    = real_part
    quat[:, 1 : ] = axis
    # Fix the failure cases
    quat[mask_parallel]      = identity_quats
    quat[mask_anti_parallel] = flip_quat
    # Normalize the quat
    quat = F.normalize(quat, p=2, dim=1)
    return quat

def rotm_rot_from_v1_to_v2(v1, v2):
    """
    Generate a rotation matrix that moves v1 to v2
    """
    B = v1.shape[0]
    # Normalize inputs
    v1 = F.normalize(v1, p=2, dim=1)
    v2 = F.normalize(v2, p=2, dim=1)
    axis = torch.cross(v1, v2) # v [B x 3]
    # Compute angles (c = cosine of angle)
    c = torch.bmm( v1.unsqueeze(1), v2.unsqueeze(-1) ).squeeze(-1).squeeze(-1)
    c_mask = (c < -0.99) # B
    # Compute the cross product operator matrix for v
    axis_x = cross_prod_operator_matrix(axis) # B x 3 x 3
    axis_x_sq = torch.bmm(axis_x, axis_x)
    # Identity matrix
    idenmat = torch.eye(3).unsqueeze(0).expand(B,-1,-1)
    # Squared correction coef 
    coef = 1.0 / (1.0 + c)
    # Apply Rodriguez formula
    R = idenmat + axis_x + coef * axis_x_sq
    # Deal with anti-parallel case
    R[c_mask] = torch.tensor([ [-1.0, 0.0,  0.0], 
                               [ 0.0, 1.0,  0.0], 
                               [ 0.0, 0.0, -1.0]])
    return R

def cross_prod_operator_matrix(axis):
    axis_x = torch.zeros(B, 3, 3)
    axis_x[:, 0, 1] = - axis[2]
    axis_x[:, 1, 0] = axis[2]
    axis_x[:, 0, 2] = axis[1]
    axis_x[:, 2, 0] = - axis[1]
    axis_x[:, 1, 2] = - axis[0]
    axis_x[:, 2, 1] = axis[0]
    return axis_x

def random_upper_hemi_TB_to_z_via_quats(B):
    """
    Sample B points in the upper hemisphere.
    Compute the rotation of each to "target" (usually towards the camera).
    Convert this to Tait-Bryan angles [B x 3]
    
    May introduce twists.
    """
    azimuths   = torch.rand(B) * 2 * np.pi
    elevations = torch.rand(B) * np.pi / 2
    cartesian_points = unit_spherical_rads(azimuths, elevations)
    # Target to rotate towards.
    z_hat       = torch.zeros(B, 3)
    z_hat[:, 2] = 1.0
    # Get rotation quaternions
    q = quat_rot_from_v1_to_v2(cartesian_points, z_hat)
    # Get TB angles
    return quat_to_tait_bryan(q)

def random_upper_hemi_rotM_rod(B):
    """
    Use the Rodriguez rotation matrix generation formula to compute a random rotation
        that moves an upper hemi vector to (0,0,1).

    May introduce twists.

    Output R [B x 3 x 3]
    """
    azimuths   = torch.rand(B) * 2 * np.pi
    elevations = torch.rand(B) * np.pi / 2
    cartesian_points = unit_spherical_rads(azimuths, elevations)
    # Target to rotate towards.
    z_hat       = torch.zeros(B, 3)
    z_hat[:, 2] = 1.0
    return rotm_rot_from_v1_to_v2(cartesian_points, z_hat)

def random_upper_hemi_rotm_manual(B, fixed_azi=None, fixed_elev=None):
    """
    Strat: 
        (1) rotate randomly in azimuth (about y-axis) [0,2pi]
        (2) randomly rotate y_hat=j_hat towards the camera 
            (@ positive z axis) [0,pi/2]
    """
    # Azimuthal rotation
    if fixed_azi is None:
        azi_angles = torch.rand(B) * math.pi * 2.0
    else:
        azi_angles = torch.ones(B) * fixed_azi
    sa = torch.sin(azi_angles)
    ca = torch.cos(azi_angles)
    R_1s = torch.zeros(B,3,3)
    R_1s[:, 0, 0] = ca 
    R_1s[:, 0, 2] = sa 
    R_1s[:, 1, 1] = 1.0 
    R_1s[:, 2, 0] = -sa 
    R_1s[:, 2, 2] = ca 
    # Elevation change (rotate the y-axis towards the z-axis)
    # The key is to use the axis-angle representation and choose to rotate
    #   about the x-axis.
    # We don't multiply by -1 so that y_hat goes towards z_hat, instead of away from it.
    if fixed_elev is None:
        elev_angles = torch.rand(B) * math.pi / 2.0 # * -1.0
    else:
        elev_angles = torch.ones(B) * fixed_elev
    se = torch.sin(elev_angles)
    ce = torch.cos(elev_angles)
    R_2s = torch.zeros(B, 3, 3)
    R_2s[:, 0, 0] = 1.0 
    R_2s[:, 1, 1] = ce 
    R_2s[:, 1, 2] = -se
    R_2s[:, 2, 1] = se
    R_2s[:, 2, 2] = ce 
    # The final rotation is their composition R = R_2 R_1
    # Make sure this is applied by LEFT multiplication
    return torch.bmm(R_2s, R_1s)


def sample_uniform_rotation_quat(B):
    """
    We follow the Shoemake (III.6 - Uniform Random Rotations) algorithm:
        x0, x1, x2 ~ U[0,1]
        theta1 = 2 pi x1
        theta2 = 2 pi x2
        s1 = sin(theta1)
        s2 = sin(theta2)
        c1 = cos(theta1)
        c2 = cos(theta2)
        r1 = sqrt( 1 - x0 )
        r2 = sqrt( x0 )
        q  = < s1 r1, c1 r1, s2 r2, c2 r2 > // unit quat
        // Note: [qimg1, qimg2, qimg3, q_real] = q 
        // BUT we output [q_real, qimg1, qimg2, qimg3]
    """
    eps = 1e-5
    X  = torch.rand(B,3)
    x0 = X[:, 0].clamp(min=eps, max=1.0-eps)
    x1 = X[:, 1]
    x2 = X[:, 2]
    theta1 = 2 * math.pi * x1
    theta2 = 2 * math.pi * x2
    s1 = torch.sin(theta1)
    s2 = torch.sin(theta2)
    c1 = torch.cos(theta1)
    c2 = torch.cos(theta2)
    r1 = torch.sqrt(1.0 - x0)
    r2 = torch.sqrt(x0)
    return torch.stack( (c2*r2, s1*r1, c1*r1, s2*r2), dim = 1)


def two_quats_to_isoclinic_rotations(qL, qR):
    """
    Generates a batch of 4D rotations matrices from two quaternion pairs.
    The quats need not be normalized (it will be done within the method).

    Inputs:
        qL: B x 4
        qR: B x 4
    Returns:
        R: B x 4 x 4

    It appears that both Kayley and Van Elfrinkhof found out how to decompose a 4D rotation
        matrix R into a product of two isoclinic rotation matrices (RL and RR), which can be viewed as each being
        equivalent to a versor (unit quaternion). RL and RR are actually commutative, as matrices. Thus,
        R = RR RL = RL RR, so that rotating v in R^4 is done via R v. As quaternions, if L and R are the quaternions
        generating RL and RR, we perform L v R (as quat multiplication). 

    References:
        Perez-Gracia and Thomas, On Cayley's Factorization of 4D Rotations and Applications, 2017.
        Mebius, A Matrix-based proof of the quaternion representation theorem for four-dimensional rotations, 2004.

    Aside 1: there are 6 parameters for 4D rotations. Using Euler angles, for instance, requires six angles
        (and thus 6 basic matrices) to parameterize the full 4D rotation space. For this reason, the
        isoclinic decomposition seems simpler.
    Aside 2: these formulas are very different than for quat -> mat in 3d because in that case we are 
        assembling a single matrix that performs R_q v = q v q*, for v being treated as a purely imaginary quaternion.

    """
    B = qL.shape[0]
    # Normalize quats into versors
    qL = F.normalize(qL, p=2, dim=1)
    qR = F.normalize(qR, p=2, dim=1)
    # Disassmble the quaternions
    l0 = qL[:, 0]
    l1 = qL[:, 1]
    l2 = qL[:, 2]
    l3 = qL[:, 3]
    r0 = qR[:, 0]
    r1 = qR[:, 1]
    r2 = qR[:, 2]
    r3 = qR[:, 3]
    # Assemble left and right isoclinic rotation matrices
    # Notice the rows are an orthonormal basis by construction
    M_L = torch.stack([  l0, -l3,  l2, -l1, 
                         l3,  l0, -l1, -l2,
                        -l2,  l1,  l0, -l3,
                         l1,  l2,  l3,  l0], dim=1).view(B,4,4)
    M_R = torch.stack([  r0, -r3,  r2, r1, 
                         r3,  r0, -r1, r2,
                        -r2,  r1,  r0, r3,
                        -r1, -r2, -r3, r0], dim=1).view(B,4,4)
    R = torch.bmm( M_L, M_R )
    return R

class RotationalTransformer4D(nn.Module):
    """
    Map a random input unit quat v to a 4D rotation R, and use that to compute the output rotation
        u = R v as a versor. 
    """
    def __init__(self, map_4d_to_8d):
        super(RotationalTransformer4D, self).__init__()
        # Mapping from a 4D vector (one quat) to an 8D vector (two quats)
        self.g = map_4d_to_8d

    def forward(self, q):
        """
        Args:
            q: B x 4 (input uniformly random quats; i.e., xi_p)

        Returns:
            theta: B x 3 (TB angles)
        """
        qLR = self.g(q) # Generate 8d vector from the uniformly random rotation (B x 8)
        R = two_quats_to_isoclinic_rotations(qLR[:, 0:4], qLR[:, 4:]) # B x 4 x 4 
        Rq = torch.bmm(R, q.unsqueeze(-1)) # Rq = R(q) q, rotational transform [B x 4]
        tb_theta = quat_to_tait_bryan(q) # B x 3 [roll, pitch, yaw]
        return tb_theta


class RotationalTransformBasedEuclideanTransformGenerator(nn.Module):
    """
    Map from (u, xi_p_translation) to (r,t).
    I.e., map from the uniformly random versor and the latent translation to
        the intermediate rotation representation and the real translation.
    """
    def __init__(self, map_4d_to_8d, map_xi_p_trans_to_t):
        super(RotationalTransformBasedEuclideanTransformGenerator, self).__init__()
        self.rotation_decoder = RotationalTransformer4D(map_4d_to_8d)
        self.translation_decoder = map_xi_p_trans_to_t

    def forward(self, xi_p):
        """
        Assume xi_p = (u, xi_p_translation) [B x (4+dim(xi_p_t))]
        Returns (r, t)
        """
        u_q    = xi_p[:, 0:4]
        xi_p_t = xi_p[:, 4:]
        r = self.rotation_decoder(u_q) # TB angles [B x 3]
        t = self.translation_decoder(xi_p_t) # Raw translation [B x 3]
        return torch.cat( (r,t), dim = 1 )


def target_vertex_projected_locations(V, ren, targets, in_unnorm_pixel_coords=False, img_sl=None, as_pil_type="point"):
    """
    Determines where in an image a list of targeted vertices will fall.
    
    Args:
        V : B x |V| x 3
        ren: soft renderer 
        targets: N_T (vertex indices of interest)

    Returns:
        The locations of the N_T vertices in each batch member mesh in an image
        rendered by the input renderer (B x N_T x 2).
        If as_pil_circs, we output a [B x N_T x 4] output instead.
    """
    assert len(V.shape) == 3, str(V.shape) # B x |V| x 3
    Vt = ren.renderer.transform.transformer(V)
    # Note that the third channel is the depth (z in camera coords)
    U = Vt[:,:,0:2] # x,y coordinates in pre-discretized img space
    U[:,:,1] = -1.0 * U[:,:,1] # Flip only the y-axis [B x |V| x 2]
    # Select only desired vertices
    proj_targs = U[:, targets, :]
    if in_unnorm_pixel_coords:
        S = img_sl
        # Map from [-1,1] to [0,S]
        proj_targs = (proj_targs + 1.0) * S / 2.0
    if as_pil_type == "point":
        pass # Already done
    elif as_pil_type == 'ellipse':
        proj_targs = to_ellipse_specification(proj_targs)
    else:
        raise ValueError('Unknown pil drawing type')
    return proj_targs # x, y per vertex (per batch)


def to_ellipse_specification(C, radius=2):
    """
    C: B x nT x 2 (circle centers)

    Takes a batch of target point sets, returns a B x nT x 4 ellipse spec as output.
    Spec: (x1, y1, x2, y2), with {x,y}1 < {x,y}2.
    """
    C_mins = C - radius 
    C_maxs = C + radius 
    a = torch.zeros(C.shape[0], C.shape[1], 4)
    a[:, :, 0:2] = C_mins
    a[:, :, 2: ] = C_maxs
    return a

def pixel_unproject(V, ren, I):
    """
    Pixel Unprojection Function.

    Args:
        V: B x |V| x 3
        ren: object with renderer field
        I: B x C x H x W

    Returns:
        P: values from the projected vertex per pixel (B x |V| x C)
    """
    assert len(V.shape) == 3, str(V.shape) # B x |V| x 3
    assert len(I.shape) == 4, str(I.shape) # B x C x H x W
    assert V.shape[0]   == I.shape[0]
    Vt = ren.renderer.transform.transformer(V)
    # Note that the third channel is the depth (z in camera coords)
    U = Vt[:,:,0:2] # x,y coordinates in pre-discretized img space
    U[:,:,1] = -1.0 * U[:,:,1]
    G = U.unsqueeze(2) # B x |V| x 1 x 2
    out = F.grid_sample(input = I, grid  = G, align_corners = True)
    out = out.squeeze(-1).permute(0,2,1) # B x |V| x C
    return out


def get_pixel_unprojection_with_vertex_occlusion(V, ren, I, faces, 
                                                 occlusion_threshold = 0.05,
                                                 eps = 1e-5,
                                                 gs_mode = 'b', # 'n', # 'b',
                                                 gs_boundary_padding = 'border',
                                                 align_corners = True,
                                                 detach_occ_info = False
                                                 ):
    """
    Performs pixel unprojection onto the nodes of a given mesh.

    TODO: no handling for vertices being outside of the view frustum
        Currently they are not considered occluded.
        They are assigned pixels by projection outward of the boundary by default.

    Args:
        V: vertices of deformed templates (B x |V| x 3)
        ren: Softras renderer object
        I: image batch (B x 4 x H x W)
        faces: mesh faces 
        gs_mode: 'nearest'/'n' or 'bilinear'/'b' mode for grid_sampling
        align_corners: setting for the grid_sampling

    Returns:
        pixel_value_per_node: B x |V| x 4 (unprojected pixel values per node)
        rendered_depth_alpha_mask: B x 1 x H x W (alpha channel from depth render)
        depth_image: B x 1 x H x W (single-channel depth image render of (V,F))
        depth_alpha_mask_value_per_node: B x |V| (unproj of depth render alpha channel)
        unproj_zbuffer_value_per_node: B x |V| (unprojected depth_img/z_buffer value)
        depth_per_v: B x |V| (true_depth per node)
        depth_difference: B x |V| (true_depth - z_buffer_unprojection)
        occlusion_signal: B x |V| (detached)
    """
    # Checks
    assert len(V.shape)     == 3, str(V.shape) # B x |V| x 3
    assert len(faces.shape) == 3, str(faces.shape) # B x |F| x 3
    assert len(I.shape)     == 4, str(I.shape) # B x C x H x W
    assert I.shape[1]       == 4 
    if gs_mode.lower().strip() == 'b': gs_mode = 'bilinear'
    if gs_mode.lower().strip() == 'n': gs_mode = 'nearest'
    B, nV, _ = V.shape
    B, C, H, W = I.shape
    device = V.device

    #print('ttt', V.device, I.device, ren.at.device, ren.up.device)
    #print('uuu', V.device, ren.renderer.transform.transformer._at)
    ren.renderer.transform.transformer._at = ren.renderer.transform.transformer._at.to(device)
    ren.renderer.transform.transformer._up = ren.renderer.transform.transformer._up.to(device)

    ### Transform into camera coordinates ###
    # Note that the third channel is the depth (z in camera coords)
    Vt = ren.renderer.transform.transformer(V)
    # Get position in image coordinates
    U = Vt[:,:,0:2] # x,y coordinates in pre-discretized img space
    U[:,:,1] = -1.0 * U[:,:,1] # Flip y-axis

    with autograd.set_grad_enabled(not detach_occ_info):
        ### Get depth per vertex (z-value in camera coordinates) ###
        depth_per_v = Vt[:,:,2] # .unsqueeze(-1).expand(-1,-1,3) # B x |V|
        depth_per_v_min = depth_per_v.min(dim = 1)[0].unsqueeze(-1) # B x 1
        depth_per_v_max = depth_per_v.max(dim = 1)[0].unsqueeze(-1) # B x 1
        depth_per_v_range = depth_per_v_max - depth_per_v_min + eps
        depth_per_v_normalized = 2.0 * ( (depth_per_v - depth_per_v_min) 
                                         / depth_per_v_range ) - 1.0
        
        ### Render depth image ###
        # Depth texture (B x |V| x C=1) -> gets expanded to C = 3 by softras
        rendered_depth = ren( V = V, 
                              F = faces, 
                              T = depth_per_v_normalized.unsqueeze(-1).expand(-1,-1,3)
                             ) # B x (3+1) x H x W
        depth_image = rendered_depth[:, 0:3, :, :]
        rendered_depth_alpha_mask = rendered_depth[:, -1, :, :].unsqueeze(1)
    
        ### Depth image denormalization ###
        # De-normalize the depths (undo the [-1,1] normalization).
        # We do this so that the values are comparable to the real depth (which we
        #   got from the camera transformation).
        # TODO NOTE why no rendering post-processing needed? Just the denorm suffices?
        # We only need one channel from the depth image (all are the same)
        depth_image = depth_image[:,0,:,:].unsqueeze(1)
        #print('pre-depth image minmax', 
        #      depth_image.min(-1)[0].min(-1)[0],
        #      depth_image.max(-1)[0].max(-1)[0])
        #print('uuu', depth_per_v_min.shape, depth_per_v_range.shape, depth_image.shape)
        depth_image = ( depth_per_v_min.view(B,1,1,1) + 
                        ( depth_image + 1.0 ) * depth_per_v_range.view(B,1,1,1) / 2.0 )
        #print('post-depth image minmax', 
        #      depth_image.min(-1)[0].min(-1)[0],
        #      depth_image.max(-1)[0].max(-1)[0])
        
        # Mask out the depth image background to be zero-valued
        # Warning: the bg is set to zero, which gets denormalized. 
        # Often, zero means a plane ~halfway through the object, potentially near the
        #   back of the *visible* vertices of the object.
        # Hence, the bg may be assigned a value similar to the furthest back pixels in
        #   the depth image. However, nothing guarantees this and it is an unstable
        #   and dangerous thing to assume.
        # Hence why we mask out the background to be zero-valued, and then
        #   renormalize the contrast after unprojection by dividing out the alpha value.
        depth_image = depth_image * rendered_depth_alpha_mask 

    ### Perform unprojection via grid-sampling ###
    # Construct sampling grid based on x,y camera coordinates of vertices
    G = U.unsqueeze(2) # B x |V| x 1 x 2 [x,y coords of projected verts to sample]
    # Perform grid sample of (pixel colours [4] + depths [1] + depth-render alpha [1])
    out = F.grid_sample(
            input = torch.cat( 
                      (I, depth_image, rendered_depth_alpha_mask), 
                      dim = 1), 
            grid = G, align_corners = align_corners, 
            mode = gs_mode, padding_mode = gs_boundary_padding)
    # We now have the unprojected colour per pixel, along with the depth image 
    #   value per node, essentially read from the z-buffer
    # This is, of course, different from the depth (z in camera coordinates),
    #   for occluded targets (nodes).
    out = out.squeeze(-1).permute(0,2,1) # B x |V| x (C + 1)
    # Unpack the unprojected node values
    pixel_value_per_node            = out[:, :, 0:4] # Unprojections from I [incl. alpha]
    unproj_zbuffer_value_per_node   = out[:, :, 4]   # Unprojection from depth image 
    depth_alpha_mask_value_per_node = out[:, :, 5]   # Unprojection from depth image mask
    # Renormalize the contrast of the depth value via the rendered mask.
    # This overall means we did grid_sample(mask * depth) / grid_sample(mask).
    unproj_zbuffer_value_per_node = ( unproj_zbuffer_value_per_node / 
                                      depth_alpha_mask_value_per_node.clamp(min=eps) )
    ### Compute the occlusion values via unprojected depth difference ###
    # d = depth difference = true depth - unproj depth 
    # if d == 0, then the camera depth is the unproj depth, so the vertex is
    #   not occluded. if d > 0, then true depth is farther behind than the unprojection,
    #   so the vertex is occluded.
    depth_difference = depth_per_v - unproj_zbuffer_value_per_node
    if detach_occ_info:
        depth_difference = depth_difference.detach()
    # Decided whether a vertex is occluded based on the threshold
    occluded_cells = (depth_difference > occlusion_threshold)
    # The occlusion signal is a bit that signifies whether the target is occluded
    occlusion_signal = occluded_cells.long().float().detach()
    
    return (pixel_value_per_node, # RGBA pixel value unprojected from input image
            rendered_depth_alpha_mask, # Alpha channel mask from rendered depth image
            depth_image, # Rendered depth image (scalar image)
            depth_alpha_mask_value_per_node, # Unprojection of depth render alpha mask
            unproj_zbuffer_value_per_node, # Unprojection of depth image values
            depth_per_v, # Depth value per vertex (cam coords, not unprojected)
            depth_difference, # Difference between true z-depth and depth image unproj
            occlusion_signal) # One for occluded verts, zero otherwise



# Gives a set of random, highly different colors
# starting_colors is a list of 3D numpy arrays
# Output is a list of 3D numpy arrays
# e.g. see https://gist.github.com/adewes/5884820
def generateDifferentRandomColors(desired_length, 
                                  starting_colors = [ np.array([0.0, 0.0, 0.0]) ], 
                                  num_samples = 250, 
                                  as_ints = False):
    if starting_colors is None: colors = []
    else: colors = starting_colors
    _eps = 0.00001
    sqdist = lambda u,v: np.inner(u-v, u-v)
    while len(colors) < desired_length:
        tries = npr.uniform(low=_eps, high=1-_eps, size=(num_samples, 3))
        #min_dist_to_any_curr_color = lambda r:  np.minimum( [ sqdist(c,r) for c in colors ] )
        min_dist_to_any_curr_color = lambda r:  min( [ sqdist(c,r) for c in colors ] )
        min_dists = [ min_dist_to_any_curr_color(tries[i,:]) for i in range(num_samples) ]
        ind_of_max_dist = np.argmax(min_dists)
        # Choose the max color and add it to the chosen colors
        colors.append( tries[ind_of_max_dist,:] )
    if as_ints:
        colors = [ tuple( (color * 255).astype(int) ) for color in colors ]
    return colors 


def okabe_ito_colours():
    return [
        (193, 126, 165),
        (199, 100, 38),        
        (109, 179, 228),
        (69,  155, 118),
        (239, 227, 98),
        (46,  114, 173),
        (220, 161, 56),
        (0,   0,   0),
    ]






#
