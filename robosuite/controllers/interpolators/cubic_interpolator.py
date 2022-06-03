import numpy as np

import robosuite.utils.transform_utils as T
from robosuite.controllers.interpolators.base_interpolator import Interpolator


class CubicInterpolator(Interpolator):
    def __init__(
        self,
        ndim: int,
        controller_freq: float,
        policy_freq: float,
        ramp_ratio: float = 0.2,
        use_delta_goal: bool = False,
        ori_interpolate=None,
    ):
        self.dim = ndim  # Number of dimensions to interpolate
        self.ori_interpolate = ori_interpolate  # Whether this is interpolating orientation or not
        self.order = 3  # Order of the interpolator (3 = cublic spline)
        self.step = 0  # Current step of the interpolator
        self.total_steps = np.ceil(
            ramp_ratio * controller_freq / policy_freq
        )  # Total num steps per interpolator action
        self.use_delta_goal = use_delta_goal  # Whether to use delta or absolute goals (currently
        # not implemented yet- TODO)
        self.set_states(dim=ndim, ori=ori_interpolate)

    def set_states(self, dim=None, ori=None):
        """
        Updates self.dim and self.ori_interpolate.

        Initializes self.start and self.goal with correct dimensions.

        Args:
            ndim (None or int): Number of dimensions to interpolate

            ori_interpolate (None or str): If set, assumes that we are interpolating angles (orientation)
                Specified string determines assumed type of input:

                    `'euler'`: Euler orientation inputs
                    `'quat'`: Quaternion inputs
        """
        # Update self.dim and self.ori_interpolate
        self.dim = dim if dim is not None else self.dim
        self.ori_interpolate = ori if ori is not None else self.ori_interpolate

        # Set start and goal states
        if self.ori_interpolate is not None:
            raise NotImplementedError
            # if self.ori_interpolate == "euler":
            #     self.start = np.zeros(3)
            # else:  # quaternions
            #     self.start = np.array((0, 0, 0, 1))
        else:
            self.start_pos = np.zeros(self.dim)
            self.start_vel = np.zeros(self.dim)
            self.start_acc = np.zeros(self.dim)

        self.goal_pos = np.array(self.start_pos)
        self.goal_vel = np.array(self.start_vel)
        self.goal_acc = np.array(self.start_acc)

    def set_goal(self, goal_pos, goal_vel=None, goal_acc=None):
        """
        Takes a requested (absolute) goal and updates internal parameters for next interpolation step

        Args:
            np.array: Requested goal (absolute value). Should be same dimension as self.dim
        """
        # First, check to make sure requested goal shape is the same as self.dim
        if goal_pos.shape[0] != self.dim:
            print("Requested goal_pos: {}".format(goal_pos))
            raise ValueError(
                "LinearInterpolator: Input size wrong for goal_pos; got {}, needs to be {}!".format(
                    goal_pos.shape[0], self.dim
                )
            )

        # Update start and goal
        self.start_pos = np.array(self.goal_pos)
        self.goal_pos = np.array(goal_pos)

        if goal_vel is not None:
            self.start_vel = np.array(self.goal_vel)
            self.goal_vel = goal_vel

        if goal_acc is not None:
            self.start_acc = np.array(self.goal_acc)
            self.goal_acc = goal_acc

        # Calculate interpolation coeffs
        self._calculate_interpoltion_coeffs()

        # Reset interpolation steps
        self.step = 0

    def _calculate_interpoltion_coeffs(self):
        a0 = self.start_pos
        a1 = self.start_vel
        a2 = 0.5 * self.start_acc

        inv_t = np.array(
            [[10.0, -4.0, 0.5], [-15.0, 7.0, -1.0], [6.0, -3.0, 0.5]]
        )  # pre-inverted matrix with t_0 = 0 and t_1 = 1

        remaining_coeffs = inv_t.dot(
            np.array([self.goal_pos - (a0 + a1 + a2), self.goal_vel - (a1 + 2 * a2), self.goal_acc - 2 * a2])
        )
        self.interp_coeffs = np.column_stack((a0, a1, a2, remaining_coeffs.T))

    def get_interpolated_goal(self):
        """
        Provides the next step in interpolation given the remaining steps.

        NOTE: If this interpolator is for orientation, it is assumed to be receiving either euler angles or quaternions

        Returns:
            np.array: Next position in the interpolated trajectory
        """

        # Calculate the desired next step based on remaining interpolation steps
        if self.ori_interpolate is not None:
            raise NotImplementedError
        else:
            s = self.step / self.total_steps
            x_pos = self.interp_coeffs.dot(np.array([1, s, s ** 2, s ** 3, s ** 4, s ** 5]))
            x_vel = self.interp_coeffs[:, 1:].dot(np.array([1, 2 * s, 3 * s ** 2, 4 * s ** 3, 5 * s ** 4]))
            x_acc = self.interp_coeffs[:, 2:].dot(np.array([2, 2 * s, 12 * s ** 2, 20 * s ** 3]))

        # Increment step if there's still steps remaining based on ramp ratio
        if self.step < self.total_steps - 1:
            self.step += 1

        # Return the new interpolated step
        return x_pos, x_vel, x_acc
