# InEKF, OpenVINS-InEKF, and RGB-D-InEKF Fusion

Date: 2026-05-06

This note documents the estimator formulation used in the current ROS2 experiments, the OpenVINS/InEKF fusion cycle used during earlier debugging, and the current D455 RGB-D/InEKF fusion mode. The current stable RGB-D frontend uses OpenCV `cv::rgbd::RgbdOdometry`; the estimator-side fusion uses the `icra_oc_xy` position update in `scripts/inekf_analog_node.py`.

## 1. InEKF Formulation

The state used by the analog InEKF node is

$$
x =
\left(
R_{WB},\,
v_W,\,
p_W,\,
d_{W,1},\ldots,d_{W,4},\,
b_g,\,
b_a
\right),
$$

where `R_WB` is the body orientation in the world frame, `v_W` and `p_W` are body velocity and position, `d_W,i` are world-frame contact foot positions, and `b_g,b_a` are gyro and accelerometer biases.

The IMU propagation subtracts bias and integrates the body-frame measurements:

$$
\omega_B = \omega_m - b_g,
\qquad
a_B = a_m - b_a .
$$

The continuous-time model is

$$
\dot{R}_{WB} = R_{WB}[\omega_B]_\times,
$$

$$
\dot{v}_W = R_{WB}a_B + g_W,
$$

$$
\dot{p}_W = v_W.
$$

The discrete propagation implemented at each IMU sample is

$$
p_{k+1} = p_k + v_k \Delta t + \frac{1}{2}
\left(R_k(a_m-b_a)+g_W\right)\Delta t^2,
$$

$$
v_{k+1} = v_k + \left(R_k(a_m-b_a)+g_W\right)\Delta t,
$$

$$
R_{k+1} = R_k \exp\left((\omega_m-b_g)\Delta t\right).
$$

Here the implementation uses gravity vector `[0, 0, -g]^T`.

The covariance is propagated with the linearized error-state model:

$$
P_{k+1} = F_k P_k F_k^T + G_k Q_k G_k^T.
$$

The implemented error state is ordered approximately as

$$
\delta x =
\left(
\delta \theta,\,
\delta v,\,
\delta p,\,
\delta d_1,\ldots,\delta d_4,\,
\delta b_g,\,
\delta b_a
\right).
$$

Contacts provide the leg odometry constraints. For each trusted contact foot, forward kinematics gives the foot position and velocity in the body frame. Assuming the contact point is stationary, the body velocity estimate from one contact is

$$
v_{B,i}^{leg} = -\left(\omega_B \times r_{B,i} + \dot{r}_{B,i}\right).
$$

Multiple contact estimates are averaged with contact weights. The resulting body-frame leg velocity is converted to world frame and fused as a velocity measurement:

$$
r_v = R_{WB}v_B^{leg} - v_W,
$$

$$
H_v =
\begin{bmatrix}
0_{3\times3} & I_3 & 0_{3\times3} & \cdots
\end{bmatrix}.
$$

The Kalman update is the standard error-state update:

$$
S = HPH^T + R,
\qquad
K = PH^T S^{-1},
$$

$$
\delta x = Kr,
$$

$$
P^+ = (I-KH)P(I-KH)^T + KRK^T.
$$

The state is then corrected by applying the error-state increment to orientation, velocity, position, contact foot states, and biases.

## 2. OpenVINS-InEKF Fusion Cycle

The OpenVINS fusion path was implemented as a generic visual-odometry feedback loop. In the code it is still named `openvins_*`, but the same callback path is also reused by the RGB-D VO frontend because both publish odometry-style pose messages.

At runtime the cycle is:

1. InEKF propagates at IMU rate.
2. OpenVINS publishes a lower-rate visual-inertial odometry pose.
3. The pose is converted into the requested feedback frame: `imu`, `body`, or `base`.
4. The measurement is queued by timestamp.
5. When the InEKF IMU propagation reaches that timestamp, the latest valid visual measurement is processed.
6. The visual measurement is interpreted either as an absolute pose or as a relative pose delta.
7. A Kalman residual is built and gated by timestamp, correction magnitude, rate, quality, and NIS.
8. If accepted, the InEKF state and covariance are updated.

The relative-mode formulation stores a reference visual pose and the matching InEKF state at time `t_i`. When a new visual pose arrives at `t_j`, the visual delta is

$$
\Delta p_{VO} = p_{VO,j} - p_{VO,i},
\qquad
\Delta R_{VO} = R_{VO,i}^{T}R_{VO,j}.
$$

The target pose predicted from the old InEKF state plus the visual relative motion is

$$
p_{target} =
p_{I,i} + R_{I,i}\Delta p_{VO},
$$

$$
R_{target} =
R_{I,i}\Delta R_{VO}.
$$

The position residual is

$$
r_p = p_{target} - p_{I,j}.
$$

If orientation feedback is enabled, the orientation residual is

$$
r_R =
\log\left(R_{target}R_{I,j}^{T}\right).
$$

For the generic OpenVINS EKF mode, the stacked residual can include orientation, position, velocity, and delta-velocity:

$$
r =
\begin{bmatrix}
r_R \\
r_p \\
r_v
\end{bmatrix}.
$$

The corresponding measurement Jacobian is assembled over the error-state. For a position residual, the code uses the standard position block

$$
H_p =
\begin{bmatrix}
-[p_W]_\times & 0_{3\times3} & I_3 & 0 & \cdots
\end{bmatrix}
$$

unless the mode is absolute trajectory alignment, where the orientation coupling is disabled.

The measurement covariance is taken from the visual odometry covariance when available, floored by configured noise values, and then inflated by consistency or quality gates:

$$
R_p =
\operatorname{sanitize}
\left(
R_{VO,p} + R_{anchor}
\right).
$$

The OpenVINS loop was useful for debugging frame alignment, timestamp synchronization, and residual construction. The main limitation observed in the June data was that OpenVINS/mono visual drift did not reliably beat contact-aided InEKF; fusing it too strongly often made position worse.

## 3. RGB-D-InEKF Fusion Formulation

The current stable RGB-D frontend uses D455 color, aligned depth, and camera intrinsics. The launcher is

```bash
scripts/run_june23_d455_rgbd_icra_oc_bag.sh
```

with the key settings:

```text
use_rgbd_depth:=true
topic_depth0:=/go2/d455/depth/image_rect_raw
rgbd_vo_impl:=cpp
rgbd_vo_nidevo_backend:=opencv_rgbd
rgbd_vo_window_filter:=true
rgbd_vo_filter_window_s:=0.60
rgbd_vo_filter_min_dt:=0.20
openvins_feedback_mode:=icra_oc_xy
openvins_measurement_mode:=relative
openvins_alignment_mode:=yaw
openvins_feedback_use_position:=true
openvins_feedback_use_orientation:=false
openvins_feedback_use_velocity:=false
openvins_feedback_use_absolute_z:=false
```

The RGB-D visual frontend creates OpenCV RGB-D odometry as

```cpp
cv::rgbd::RgbdOdometry::create(
    camera_matrix,
    min_depth,
    max_depth,
    max_depth_diff,
    iter_counts,
    min_gradients,
    max_points_part,
    cv::rgbd::Odometry::RIGID_BODY_MOTION);
```

Then each step calls

```cpp
opencv_odom_->compute(
    prev_gray,
    prev_depth,
    cv::Mat(),
    gray,
    curr_depth,
    cv::Mat(),
    rt_curr_prev);
```

This estimates an RGB-D relative rigid transform between two camera frames:

$$
T_{C_jC_i}^{rgbd}
=
\begin{bmatrix}
R_{C_jC_i} & t_{C_jC_i} \\
0 & 1
\end{bmatrix}.
$$

The implementation inverts this to get the previous-to-current transform:

$$
T_{C_iC_j}^{rgbd} =
\left(T_{C_jC_i}^{rgbd}\right)^{-1}.
$$

The frontend rejects obviously invalid steps using maximum rotation and translation limits:

$$
\theta(\Delta R) < \theta_{max},
\qquad
\|\Delta p\| < p_{max}.
$$

The accepted RGB-D odometry is published as a VO pose stream. A window filter then smooths the short-horizon motion and produces a more stable relative measurement.

The current stable fusion mode is observability-constrained horizontal position fusion. The visual delta is converted into a target body/world position as

$$
p_{target} =
p_{ref} + R_{ref}\Delta p_{rgbd}.
$$

The residual is

$$
r_p = p_{target} - p_{I}.
$$

But only the horizontal components are fused:

$$
r_{xy} =
\begin{bmatrix}
r_x \\
r_y
\end{bmatrix}.
$$

The update uses only the `x` and `y` position covariance block:

$$
R_{xy} =
\lambda
\begin{bmatrix}
R_{xx} & R_{xy} \\
R_{yx} & R_{yy}
\end{bmatrix},
$$

where `lambda` is the configured `openvins_icra_xy_covariance_scale` if used. The Kalman update is performed only on the position axes:

$$
S_{xy} = P_{xy,xy} + R_{xy},
$$

$$
K_{xy} = P_{xy,xy}S_{xy}^{-1},
$$

$$
\Delta p_{xy} = K_{xy}r_{xy}.
$$

Then the state position is corrected as

$$
p_x^+ = p_x + \Delta p_x,
\qquad
p_y^+ = p_y + \Delta p_y,
\qquad
p_z^+ = p_z.
$$

The contact foot states are translated by the same horizontal correction so that the contact map remains consistent with the body pose:

$$
d_i^+ = d_i + 
\begin{bmatrix}
\Delta p_x \\
\Delta p_y \\
0
\end{bmatrix}.
$$

The stable RGB-D mode deliberately does not fuse RGB-D orientation, velocity, or absolute `z`. Full `xyz` position fusion was tested, but the vertical channel was less reliable on these sequences and could inject bad height corrections.

## 4. Experiment Description

The main evaluation used June 23 D455 bags with motion-capture ground truth. The subset used for repeated ablations contains:

```text
dancing
jump_forward_seq1
jump_forward_seq2
running-seq1
running-seq2
walking-seq2
```

Each sequence was run with two estimators:

- `raw InEKF`: contact-aided InEKF without visual correction.
- `fused`: InEKF with RGB-D `icra_oc_xy` visual position correction.

The raw InEKF contact stream was corrupted in controlled ablations:

- `mask0`: no random contact corruption.
- `mask30`: randomly mask 30% of contact episodes.
- `fake10`: randomly add 10% fake contact episodes.

The purpose of these ablations is to test whether RGB-D helps when the contact model is degraded. The RGB-D frontend itself is unchanged across ablations.

Ground-truth comparison used timestamp interpolation onto the overlapping time range. Trajectories are yaw-aligned and start-position aligned for visualization. Metrics reported in the summary files are pointwise mean position error and mean body-velocity error over the overlap.

Important output artifacts:

```text
output/june23_rgbd_icra_oc_xy_subset_mask0/summary.tsv
output/june23_rgbd_icra_oc_xy_subset_mask03/summary.tsv
output/june23_rgbd_icra_oc_xy_subset_fake10/summary.tsv
output/june23_rgbd_icra_oc_xy_subset_mask_compare/normalized_mse_raw_baseline_common_sequences.png
output/june23_rgbd_icra_oc_xy_subset_mask_compare/traj_grid_all_masks.png
```

## 5. Results

### No Contact Masking

| Sequence | Fused pos mean m | Raw pos mean m | Fused vel mean m/s | Raw vel mean m/s |
|---|---:|---:|---:|---:|
| dancing | 0.134 | 0.136 | 0.047 | 0.048 |
| jump_forward_seq1 | 0.209 | 0.203 | 0.035 | 0.038 |
| jump_forward_seq2 | 0.761 | 0.741 | 0.041 | 0.040 |
| running-seq1 | 1.264 | 1.235 | 0.082 | 0.090 |
| running-seq2 | 1.469 | 1.504 | 0.115 | 0.120 |
| walking-seq2 | 0.352 | 0.348 | 0.076 | 0.078 |

Summary:

| Ablation | Pos improved | Vel improved | Both improved |
|---|---:|---:|---:|
| `mask0` | 2 / 6 | 5 / 6 | 2 / 6 |

With clean contacts, InEKF is already strong. RGB-D gives small horizontal corrections and usually keeps velocity similar or slightly better, but position gains are limited.

### 30% Contact Masking

| Sequence | Fused pos mean m | Raw pos mean m | Fused vel mean m/s | Raw vel mean m/s |
|---|---:|---:|---:|---:|
| dancing | 0.128 | 0.083 | 0.046 | 0.048 |
| jump_forward_seq1 | 0.199 | 0.224 | 0.035 | 0.034 |
| jump_forward_seq2 | 0.748 | 0.781 | 0.041 | 0.041 |
| running-seq1 | 1.266 | 1.392 | 0.082 | 0.091 |
| running-seq2 | 1.482 | 1.606 | 0.116 | 0.120 |
| walking-seq2 | 0.453 | 0.617 | 0.073 | 0.080 |

Summary:

| Ablation | Pos improved | Vel improved | Both improved |
|---|---:|---:|---:|
| `mask30` | 5 / 6 | 5 / 6 | 4 / 6 |

This is the clearest useful case. When contacts are partially missing, the horizontal RGB-D correction compensates for contact-induced position drift. Velocity remains mostly controlled by the InEKF/contact model, so the fused velocity generally stays close to or better than raw InEKF.

### 10% Fake Contacts

| Sequence | Fused pos mean m | Raw pos mean m | Fused vel mean m/s | Raw vel mean m/s |
|---|---:|---:|---:|---:|
| dancing | 0.126 | 0.105 | 0.047 | 0.048 |
| jump_forward_seq1 | 0.221 | 0.203 | 0.035 | 0.038 |
| jump_forward_seq2 | 0.757 | 0.747 | 0.041 | 0.040 |
| running-seq1 | 1.272 | 1.301 | 0.083 | 0.090 |
| running-seq2 | 1.459 | 1.489 | 0.115 | 0.119 |
| walking-seq2 | 0.454 | 0.462 | 0.095 | 0.075 |

Summary:

| Ablation | Pos improved | Vel improved | Both improved |
|---|---:|---:|---:|
| `fake10` | 3 / 6 | 4 / 6 | 2 / 6 |

Fake contacts are harder than missing contacts because the filter receives actively wrong kinematic constraints. RGB-D still improves position in the running and walking cases, but it cannot fully cancel bad contact measurements without risking over-trusting the vision frontend.

## 6. Discussion

The RGB-D fusion is most useful when the InEKF contact model is degraded but not completely wrong. In the `mask30` ablation, fused position improves in 5 of 6 sequences, and velocity improves in 5 of 6. This supports the intended role of RGB-D: provide an external metric horizontal correction when contact constraints are intermittent.

The improvement is smaller with clean contacts because raw InEKF already has strong contact constraints. In that case the RGB-D update mostly acts as a small consistency correction rather than the dominant estimator.

The `fake10` case shows the limitation. False contacts can push the InEKF with incorrect kinematic constraints. Since the current stable RGB-D fusion only corrects horizontal position and does not estimate contact validity directly, it can reduce some drift but cannot fully reject all contact failures.

The most important design decision is fusing only `x/y` position:

$$
z,\; R,\; v,\; b_g,\; b_a
\quad\text{remain controlled by IMU/contact InEKF.}
$$

This made the system more stable than full pose fusion. RGB-D depth gives metric translation, but the tested vertical correction was noisy enough that absolute-`z` updates sometimes hurt the estimator. Horizontal-only fusion is therefore the accepted stable configuration.

Overall, the current result should be presented as an RGB-D-aided contact InEKF:

- InEKF provides high-rate propagation, contact velocity, contact position consistency, attitude, and bias handling.
- RGB-D provides lower-rate metric horizontal drift correction.
- The benefit is strongest under contact masking or partial contact dropout.
- The method is not a full tightly-coupled visual-inertial-contact estimator; it is a loosely coupled RGB-D odometry measurement update on top of the contact-aided InEKF.
