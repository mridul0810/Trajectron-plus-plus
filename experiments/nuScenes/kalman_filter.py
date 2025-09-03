import numpy as np


import numpy as np


class NonlinearKinematicBicycle:
    """
    Nonlinear Kalman Filter for a kinematic bicycle model, assuming constant longitudinal speed
    and constant heading array
    """

    def __init__(self, dt, sPos=None, sHeading=None, sVel=None, sMeasurement=None):
        self.dt = dt

        # measurement matrix
        self.C = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

        # default noise covariance
        if (sPos is None) and (sHeading is None) and (sVel is None):
            # TODO need to further check
            # sPos = 0.5 * 8.8 * dt ** 2  # assume 8.8m/s2 as maximum acceleration
            # sHeading = 0.5 * dt  # assume 0.5rad/s as maximum turn rate
            # sVel = 8.8 * dt  # assume 8.8m/s2 as maximum acceleration
            # sMeasurement = 1.0
            sPos = 12 * self.dt  # assume 6m/s2 as maximum acceleration
            sHeading = 0.5 * self.dt  # assume 0.5rad/s as maximum turn rate
            sVel = 6 * self.dt  # assume 6m/s2 as maximum acceleration
        if sMeasurement is None:
            sMeasurement = 5.0
        # state transition noise
        self.Q = np.diag([sPos ** 2, sPos ** 2, sHeading ** 2, sVel ** 2])
        # measurement noise
        self.R = np.diag([sMeasurement ** 2, sMeasurement ** 2, sMeasurement ** 2, sMeasurement ** 2])

    def predict_and_update(self, x_vec_est, u_vec, P_matrix, z_new):
        """
        for background please refer to wikipedia: https://en.wikipedia.org/wiki/Extended_Kalman_filter
        :param x_vec_est:
        :param u_vec:
        :param P_matrix:
        :param z_new:
        :return:
        """

        ## Prediction Step
        # predicted state estimate
        x_pred = self._kinematic_bicycle_model_rearCG(x_vec_est, u_vec)
        # Compute Jacobian to obtain the state transition matrix
        A = self._cal_state_Jacobian(x_vec_est, u_vec)
        # predicted error covariance
        P_pred = A.dot(P_matrix.dot(A.transpose())) + self.Q

        ## Update Step
        # innovation or measurement pre-fit residual
        y_telda = z_new - self.C.dot(x_pred)
        # innovation covariance
        S = self.C.dot(P_pred.dot(self.C.transpose())) + self.R
        # near-optimal Kalman gain
        K = P_pred.dot(self.C.transpose().dot(np.linalg.inv(S)))
        # updated (a posteriori) state estimate
        x_vec_est_new = x_pred + K.dot(y_telda)
        # updated (a posteriori) estimate covariance
        P_matrix_new = np.dot((np.identity(4) - K.dot(self.C)), P_pred)

        return x_vec_est_new, P_matrix_new

    def _kinematic_bicycle_model_rearCG(self, x_old, u):
        """
        :param x: vehicle state vector = [x position, y position, heading, velocity]
        :param u: control vector = [acceleration, steering array]
        :param dt:
        :return:
        """

        acc = u[0]
        delta = u[1]

        x = x_old[0]
        y = x_old[1]
        psi = x_old[2]
        vel = x_old[3]

        x_new = np.array([[0.], [0.], [0.], [0.]])

        x_new[0] = x + self.dt * vel * np.cos(psi + delta)
        x_new[1] = y + self.dt * vel * np.sin(psi + delta)
        x_new[2] = psi + self.dt * delta
        #x_new[2] = _heading_angle_correction(x_new[2])
        x_new[3] = vel + self.dt * acc

        return x_new

    def _cal_state_Jacobian(self, x_vec, u_vec):
        acc = u_vec[0]
        delta = u_vec[1]

        x = x_vec[0]
        y = x_vec[1]
        psi = x_vec[2]
        vel = x_vec[3]

        a13 = -self.dt * vel * np.sin(psi + delta)
        a14 = self.dt * np.cos(psi + delta)
        a23 = self.dt * vel * np.cos(psi + delta)
        a24 = self.dt * np.sin(psi + delta)
        a34 = self.dt * delta

        JA = np.array([[1.0, 0.0, a13[0], a14[0]],
                       [0.0, 1.0, a23[0], a24[0]],
                       [0.0, 0.0, 1.0, a34[0]],
                       [0.0, 0.0, 0.0, 1.0]])

        return JA


def _heading_angle_correction(theta):
    """
    correct heading array so that it always remains in [-pi, pi]
    :param theta:
    :return:
    """
    theta_corrected = (theta + np.pi) % (2.0 * np.pi) - np.pi
    return theta_corrected


class PedestrianKalmanFilter:
    """
    Extended Kalman Filter with Constant Turn Rate and Acceleration (CTRA) model.
    State: [x, y, v, yaw, yaw_rate, a]
    Measurement: [x, y]
    """

    def __init__(self, dt, sPos=1.0, sVel=1.0, sYaw=0.1, sYawRate=0.1, sAcc=1.0, sMeasurement=1.0):
        self.dt = dt
        # Process noise covariance
        self.Q = np.diag([sPos**2, sPos**2, sVel**2, sYaw**2, sYawRate**2, sAcc**2])
        # Measurement noise covariance
        self.R = np.diag([sMeasurement**2, sMeasurement**2])
        # Measurement matrix (we measure position only)
        self.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0]
        ])

    def _state_transition(self, x):
        """
        Nonlinear state transition for CTRA model.
        x: [x, y, v, yaw, yaw_rate, a]
        """
        px, py, v, yaw, yaw_rate, a = x.flatten()
        dt = self.dt

        if abs(yaw_rate) > 1e-4:
            px_new = px + (v / yaw_rate) * (np.sin(yaw + yaw_rate * dt) - np.sin(yaw)) + \
                     (a / (yaw_rate**2)) * (np.sin(yaw + yaw_rate * dt) - np.sin(yaw) - yaw_rate * dt * np.cos(yaw + yaw_rate * dt))
            py_new = py - (v / yaw_rate) * (np.cos(yaw + yaw_rate * dt) - np.cos(yaw)) - \
                     (a / (yaw_rate**2)) * (np.cos(yaw + yaw_rate * dt) - np.cos(yaw) + yaw_rate * dt * np.sin(yaw + yaw_rate * dt))
        else:
            # Straight motion (yaw_rate ~ 0)
            px_new = px + v * np.cos(yaw) * dt + 0.5 * a * np.cos(yaw) * dt**2
            py_new = py + v * np.sin(yaw) * dt + 0.5 * a * np.sin(yaw) * dt**2

        v_new = v + a * dt
        yaw_new = yaw + yaw_rate * dt
        yaw_rate_new = yaw_rate
        a_new = a

        return np.array([[px_new], [py_new], [v_new], [yaw_new], [yaw_rate_new], [a_new]])

    def _jacobian_F(self, x):
        """
        Jacobian of the state transition function with respect to the state.
        """
        px, py, v, yaw, yaw_rate, a = x.flatten()
        dt = self.dt
        F = np.eye(6)

        if abs(yaw_rate) > 1e-4:
            s = np.sin(yaw)
            c = np.cos(yaw)
            s1 = np.sin(yaw + yaw_rate * dt)
            c1 = np.cos(yaw + yaw_rate * dt)

            # Partial derivatives for px
            F[0, 2] = (1 / yaw_rate) * (s1 - s)
            F[0, 3] = (v / yaw_rate) * (c1 * (1 + dt * yaw_rate) - c) + \
                      (a / (yaw_rate**2)) * (c1 * (1 + dt * yaw_rate) - c - dt * yaw_rate * s1)
            F[0, 4] = (-v / (yaw_rate**2)) * (s1 - s) + \
                      (v / yaw_rate) * (dt * c1) + \
                      (-2 * a / (yaw_rate**3)) * (s1 - s - yaw_rate * dt * c1) + \
                      (a / (yaw_rate**2)) * (dt * c1 - dt * c1)
            F[0, 5] = (1 / (yaw_rate**2)) * (s1 - s - yaw_rate * dt * c1)

            # Partial derivatives for py
            F[1, 2] = -(1 / yaw_rate) * (c1 - c)
            F[1, 3] = (v / yaw_rate) * (s1 * (1 + dt * yaw_rate) - s) + \
                      (a / (yaw_rate**2)) * (s1 * (1 + dt * yaw_rate) - s + dt * yaw_rate * c1)
            F[1, 4] = (v / (yaw_rate**2)) * (c1 - c) - \
                      (v / yaw_rate) * (dt * s1) + \
                      (2 * a / (yaw_rate**3)) * (c1 - c + yaw_rate * dt * s1) - \
                      (a / (yaw_rate**2)) * (dt * s1 - dt * s1)
            F[1, 5] = -(1 / (yaw_rate**2)) * (c1 - c + yaw_rate * dt * s1)

            # v, yaw, yaw_rate, a
            F[2, 5] = dt
            F[3, 4] = dt
        else:
            # Straight motion
            F[0, 2] = np.cos(yaw) * dt
            F[0, 3] = -v * np.sin(yaw) * dt - 0.5 * a * np.sin(yaw) * dt**2
            F[0, 5] = 0.5 * np.cos(yaw) * dt**2

            F[1, 2] = np.sin(yaw) * dt
            F[1, 3] = v * np.cos(yaw) * dt + 0.5 * a * np.cos(yaw) * dt**2
            F[1, 5] = 0.5 * np.sin(yaw) * dt**2

            F[2, 5] = dt
            F[3, 4] = dt

        return F

    def predict(self, x, P):
        x_pred = self._state_transition(x)
        F = self._jacobian_F(x)
        P_pred = F @ P @ F.T + self.Q
        return x_pred, P_pred

    def update(self, x_pred, P_pred, z):
        """
        z: measurement [x, y]
        """
        y = z.reshape(2, 1) - self.H @ x_pred
        S = self.H @ P_pred @ self.H.T + self.R
        K = P_pred @ self.H.T @ np.linalg.inv(S)
        x_new = x_pred + K @ y
        P_new = (np.eye(6) - K @ self.H) @ P_pred
        return x_new, P_new

    def predict_and_update(self, x, P, z):
        x_pred, P_pred = self.predict(x, P)
        x_new, P_new = self.update(x_pred, P_pred, z)
        return x_new, P_new
    