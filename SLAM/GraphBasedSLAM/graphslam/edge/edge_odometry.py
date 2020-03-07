# Copyright (c) 2020 Jeff Irion and contributors

r"""A class for odometry edges.

"""


import numpy as np
import matplotlib.pyplot as plt


class EdgeOdometry:
    r"""A class for representing odometry edges in Graph SLAM.

    Parameters
    ----------
    vertices : list[graphslam.vertex.Vertex]
        A list of the vertices constrained by the edge
    information : np.ndarray
        The information matrix :math:`\Omega_j` associated with the edge
    estimate : BasePose
        The expected measurement :math:`\mathbf{z}_j`

    Attributes
    ----------
    vertices : list[graphslam.vertex.Vertex]
        A list of the vertices constrained by the edge
    information : np.ndarray
        The information matrix :math:`\Omega_j` associated with the edge
    estimate : PoseSE2
        The expected measurement :math:`\mathbf{z}_j`

    """
    def __init__(self, vertex_ids, information, estimate, vertices=None):
        self.vertex_ids = vertex_ids
        self.information = information
        self.estimate = estimate
        self.vertices = vertices

    def calc_error(self):
        r"""Calculate the error for the edge: :math:`\mathbf{e}_j \in \mathbb{R}^\bullet`.

        .. math::

           \mathbf{e}_j = \mathbf{z}_j - (p_2 \ominus p_1)


        Returns
        -------
        np.ndarray
            The error for the edge

        """
        return (self.estimate - (self.vertices[1].pose - self.vertices[0].pose)).to_compact()

    def calc_jacobians(self):
        r"""Calculate the Jacobian of the edge's error with respect to each constrained pose.

        .. math::

           \frac{\partial}{\partial \Delta \mathbf{x}^k} \left[ \mathbf{e}_j(\mathbf{x}^k \boxplus \Delta \mathbf{x}^k) \right]


        Returns
        -------
        list[np.ndarray]
            The Jacobian matrices for the edge with respect to each constrained pose

        """
        return [np.dot(np.dot(self.estimate.jacobian_self_ominus_other_wrt_other_compact(self.vertices[1].pose - self.vertices[0].pose), self.vertices[1].pose.jacobian_self_ominus_other_wrt_other(self.vertices[0].pose)), self.vertices[0].pose.jacobian_boxplus()),
                np.dot(np.dot(self.estimate.jacobian_self_ominus_other_wrt_other_compact(self.vertices[1].pose - self.vertices[0].pose), self.vertices[1].pose.jacobian_self_ominus_other_wrt_self(self.vertices[0].pose)), self.vertices[1].pose.jacobian_boxplus())]

    def to_g2o(self):
        """Export the edge to the .g2o format.

        Returns
        -------
        str
            The edge in .g2o format

        """
        return "EDGE_SE2 {} {} {} {} {} ".format(self.vertex_ids[0], self.vertex_ids[1], self.estimate[0], self.estimate[1], self.estimate[2]) + " ".join([str(x) for x in self.information[np.triu_indices(3, 0)]]) + "\n"

    def plot(self, color='b'):
        """Plot the edge.

        Parameters
        ----------
        color : str
            The color that will be used to plot the edge

        """
        xy = np.array([v.pose.position for v in self.vertices])
        plt.plot(xy[:, 0], xy[:, 1], color=color)
