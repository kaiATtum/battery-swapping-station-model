"""Electric grid models module."""

import itertools
import multimethod
import natsort
import numpy as np
import opendssdirect
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg
import typing

import mesmo.config
import mesmo.data_interface
import mesmo.solutions
import mesmo.utils

logger = mesmo.config.get_logger(__name__)


class ElectricGridModel(mesmo.utils.ObjectBase):
    """Electric grid model object consisting of the index sets for node names / branch names / der names / phases /
    node types / branch types, the nodal admittance / transformation matrices, branch admittance /
    incidence matrices and DER incidence matrices.

    :syntax:
        - ``ElectricGridModel(electric_grid_data)``: Instantiate electric grid model for given
          `electric_grid_data`.
        - ``ElectricGridModel(scenario_name)``: Instantiate electric grid model for given `scenario_name`.
          The required `electric_grid_data` is obtained from the database.

    Arguments:
        electric_grid_data (mesmo.data_interface.ElectricGridData): Electric grid data object.
        scenario_name (str): MESMO scenario name.

    Attributes:
        phases (pd.Index): Index set of the phases.
        node_names (pd.Index): Index set of the node names.
        node_types (pd.Index): Index set of the node types.
        line_names (pd.Index): Index set of the line names.
        transformer_names (pd.Index): Index set of the transformer names.
        branch_names (pd.Index): Index set of the branch names, i.e., all line names and transformer names.
        branch_types (pd.Index): Index set of the branch types.
        der_names (pd.Index): Index set of the DER names.
        der_types (pd.Index): Index set of the DER types.
        nodes (pd.Index): Multi-level / tuple index set of the node types, node names and phases
            corresponding to the dimension of the node admittance matrices.
        branches (pd.Index): Multi-level / tuple index set of the branch types, branch names and phases
            corresponding to the dimension of the branch admittance matrices.
        lines (pd.Index): Multi-level / tuple index set of the branch types, branch names and phases
            for the lines only.
        transformers (pd.Index): Multi-level / tuple index set of the branch types, branch names and phases
            for the transformers only.
        ders (pd.Index): Index set of the DER names, corresponding to the dimension of the DER power vector.
        node_voltage_vector_reference (np.ndarray): Node voltage reference / no load vector.
        branch_power_vector_magnitude_reference (np.ndarray): Branch power reference / rated power vector.
        der_power_vector_reference (np.ndarray): DER power reference / nominal power vector.
        is_single_phase_equivalent (bool): Singe-phase-equivalent modelling flag. If true, electric grid is modelled
            as single-phase-equivalent of three-phase balanced system.
        node_admittance_matrix (sp.spmatrix): Nodal admittance matrix.
        node_transformation_matrix (sp.spmatrix): Nodal transformation matrix.
        branch_admittance_1_matrix (sp.spmatrix): Branch admittance matrix in the 'from' direction.
        branch_admittance_2_matrix (sp.spmatrix): Branch admittance matrix in the 'to' direction.
        branch_incidence_1_matrix (sp.spmatrix): Branch incidence matrix in the 'from' direction.
        branch_incidence_2_matrix (sp.spmatrix): Branch incidence matrix in the 'to' direction.
        der_incidence_wye_matrix (sp.spmatrix): Load incidence matrix for 'wye' DERs.
        der_incidence_delta_matrix (sp.spmatrix): Load incidence matrix for 'delta' DERs.
        node_admittance_matrix_no_source (sp.spmatrix): Nodal admittance matrix from no-source to no-source nodes.
        node_transformation_matrix_no_source (sp.spmatrix): Nodal admittance matrix from source to no-source nodes.
        der_incidence_wye_matrix_no_source (sp.spmatrix): Incidence matrix from wye-conn. DERs to no-source nodes.
        der_incidence_delta_matrix_no_source (sp.spmatrix): Incidence matrix from delta-conn. DERs to no-source nodes.
        node_voltage_vector_reference_no_source (sp.spmatrix): Nodal reference voltage vector for no-source nodes.
        node_voltage_vector_reference_source (sp.spmatrix): Nodal reference voltage vector for source nodes.
        node_admittance_matrix_no_source_inverse (sp.spmatrix): Inverse of no-source nodal admittance matrix.
    """

    timesteps: pd.Index
    phases: pd.Index
    node_names: pd.Index
    node_types: pd.Index
    line_names: pd.Index
    transformer_names: pd.Index
    branch_names: pd.Index
    branch_types: pd.Index
    der_names: pd.Index
    der_types: pd.Index
    nodes: pd.Index
    branches: pd.Index
    lines: pd.Index
    transformers: pd.Index
    ders: pd.Index
    node_voltage_vector_reference: np.ndarray
    branch_power_vector_magnitude_reference: np.ndarray
    der_power_vector_reference: np.ndarray
    is_single_phase_equivalent: bool
    node_admittance_matrix: sp.spmatrix
    node_transformation_matrix: sp.spmatrix
    branch_admittance_1_matrix: sp.spmatrix
    branch_admittance_2_matrix: sp.spmatrix
    branch_incidence_1_matrix: sp.spmatrix
    branch_incidence_2_matrix: sp.spmatrix
    der_incidence_wye_matrix: sp.spmatrix
    der_incidence_delta_matrix: sp.spmatrix
    node_admittance_matrix_no_source: sp.spmatrix
    node_admittance_matrix_source_to_no_source: sp.spmatrix
    node_transformation_matrix_no_source: sp.spmatrix
    der_incidence_wye_matrix_no_source: sp.spmatrix
    der_incidence_delta_matrix_no_source: sp.spmatrix
    node_voltage_vector_reference_no_source: sp.spmatrix
    node_voltage_vector_reference_source: sp.spmatrix
    node_admittance_matrix_no_source_inverse: sp.spmatrix

    @multimethod.multimethod
    def __init__(self, scenario_name: str):

        # Obtain electric grid data.
        electric_grid_data = mesmo.data_interface.ElectricGridData(scenario_name)

        # Instantiate electric grid model object.
        self.__init__(electric_grid_data)

    @multimethod.multimethod
    def __init__(
        self,
        electric_grid_data: mesmo.data_interface.ElectricGridData,
    ):

        # Process overhead line type definitions.
        # - This is implemented as direct modification on the electric grid data object and therefore done first.
        electric_grid_data = self.process_line_types_overhead(electric_grid_data)

        # Obtain index set for time steps.
        # - This is needed for optimization problem definitions within linear electric grid models.
        self.timesteps = electric_grid_data.scenario_data.timesteps

        # Obtain index sets for phases / node names / node types / line names / transformer names /
        # branch types / DER names.
        self.phases = pd.Index(
            np.unique(
                np.concatenate(
                    electric_grid_data.electric_grid_nodes.apply(mesmo.utils.get_element_phases_array, axis=1).values
                )
            )
        )
        self.node_names = pd.Index(electric_grid_data.electric_grid_nodes["node_name"])
        self.node_types = pd.Index(["source", "no_source"])
        self.line_names = pd.Index(electric_grid_data.electric_grid_lines["line_name"])
        self.transformer_names = pd.Index(electric_grid_data.electric_grid_transformers["transformer_name"])
        self.branch_types = pd.Index(["line", "transformer"])
        self.der_names = pd.Index(electric_grid_data.electric_grid_ders["der_name"])
        self.der_types = pd.Index(electric_grid_data.electric_grid_ders["der_type"].unique())

        # Obtain nodes index set, i.e., collection of all phases of all nodes
        # for generating indexing functions for the admittance matrix.
        # - The admittance matrix has one entry for each phase of each node in both dimensions.
        # - There cannot be "empty" dimensions for missing phases of nodes, because the matrix would become singular.
        # - Therefore the admittance matrix must have the exact number of existing phases of all nodes.
        node_dimension = int(
            electric_grid_data.electric_grid_nodes.loc[
                :, ["is_phase_1_connected", "is_phase_2_connected", "is_phase_3_connected"]
            ]
            .sum()
            .sum()
        )
        self.nodes = pd.DataFrame(None, index=range(node_dimension), columns=["node_type", "node_name", "phase"])
        # Fill `node_name`.
        self.nodes["node_name"] = pd.concat(
            [
                electric_grid_data.electric_grid_nodes.loc[
                    electric_grid_data.electric_grid_nodes["is_phase_1_connected"] == 1, "node_name"
                ],
                electric_grid_data.electric_grid_nodes.loc[
                    electric_grid_data.electric_grid_nodes["is_phase_2_connected"] == 1, "node_name"
                ],
                electric_grid_data.electric_grid_nodes.loc[
                    electric_grid_data.electric_grid_nodes["is_phase_3_connected"] == 1, "node_name"
                ],
            ],
            ignore_index=True,
        )
        # Fill `phase`.
        self.nodes["phase"] = np.concatenate(
            [
                np.repeat(1, sum(electric_grid_data.electric_grid_nodes["is_phase_1_connected"] == 1)),
                np.repeat(2, sum(electric_grid_data.electric_grid_nodes["is_phase_2_connected"] == 1)),
                np.repeat(3, sum(electric_grid_data.electric_grid_nodes["is_phase_3_connected"] == 1)),
            ]
        )
        # Fill `node_type`.
        self.nodes["node_type"] = "no_source"
        # Set `node_type` for source node.
        self.nodes.loc[
            self.nodes["node_name"] == (electric_grid_data.electric_grid["source_node_name"]), "node_type"
        ] = "source"
        # Sort by `node_name`.
        self.nodes = self.nodes.reindex(
            index=natsort.order_by_index(self.nodes.index, natsort.index_natsorted(self.nodes.loc[:, "node_name"]))
        )
        self.nodes = pd.MultiIndex.from_frame(self.nodes)

        # Obtain branches index set, i.e., collection of phases of all branches
        # for generating indexing functions for the branch admittance matrices.
        # - Branches consider all power delivery elements, i.e., lines as well as transformers.
        # - The second dimension of the branch admittance matrices is the number of phases of all nodes.
        # - Transformers must have same number of phases per winding and exactly two windings.
        line_dimension = int(
            electric_grid_data.electric_grid_lines.loc[
                :, ["is_phase_1_connected", "is_phase_2_connected", "is_phase_3_connected"]
            ]
            .sum()
            .sum()
        )
        transformer_dimension = int(
            electric_grid_data.electric_grid_transformers.loc[
                :, ["is_phase_1_connected", "is_phase_2_connected", "is_phase_3_connected"]
            ]
            .sum()
            .sum()
        )
        self.branches = pd.DataFrame(
            None, index=range(line_dimension + transformer_dimension), columns=["branch_type", "branch_name", "phase"]
        )
        # Fill `branch_name`.
        self.branches["branch_name"] = pd.concat(
            [
                electric_grid_data.electric_grid_lines.loc[
                    electric_grid_data.electric_grid_lines["is_phase_1_connected"] == 1, "line_name"
                ],
                electric_grid_data.electric_grid_lines.loc[
                    electric_grid_data.electric_grid_lines["is_phase_2_connected"] == 1, "line_name"
                ],
                electric_grid_data.electric_grid_lines.loc[
                    electric_grid_data.electric_grid_lines["is_phase_3_connected"] == 1, "line_name"
                ],
                electric_grid_data.electric_grid_transformers.loc[
                    electric_grid_data.electric_grid_transformers["is_phase_1_connected"] == 1, "transformer_name"
                ],
                electric_grid_data.electric_grid_transformers.loc[
                    electric_grid_data.electric_grid_transformers["is_phase_2_connected"] == 1, "transformer_name"
                ],
                electric_grid_data.electric_grid_transformers.loc[
                    electric_grid_data.electric_grid_transformers["is_phase_3_connected"] == 1, "transformer_name"
                ],
            ],
            ignore_index=True,
        )
        # Fill `phase`.
        self.branches["phase"] = np.concatenate(
            [
                np.repeat(1, sum(electric_grid_data.electric_grid_lines["is_phase_1_connected"] == 1)),
                np.repeat(2, sum(electric_grid_data.electric_grid_lines["is_phase_2_connected"] == 1)),
                np.repeat(3, sum(electric_grid_data.electric_grid_lines["is_phase_3_connected"] == 1)),
                np.repeat(1, sum(electric_grid_data.electric_grid_transformers["is_phase_1_connected"] == 1)),
                np.repeat(2, sum(electric_grid_data.electric_grid_transformers["is_phase_2_connected"] == 1)),
                np.repeat(3, sum(electric_grid_data.electric_grid_transformers["is_phase_3_connected"] == 1)),
            ]
        )
        # Fill `branch_type`.
        self.branches["branch_type"] = np.concatenate(
            [np.repeat("line", line_dimension), np.repeat("transformer", transformer_dimension)]
        )
        # Sort by `branch_type` / `branch_name`.
        self.branches = self.branches.reindex(
            index=natsort.order_by_index(
                self.branches.index, natsort.index_natsorted(self.branches.loc[:, "branch_name"])
            )
        )
        self.branches = self.branches.reindex(
            index=natsort.order_by_index(
                self.branches.index, natsort.index_natsorted(self.branches.loc[:, "branch_type"])
            )
        )
        self.branches = pd.MultiIndex.from_frame(self.branches)

        # Obtain index sets for lines / transformers corresponding to branches.
        self.lines = self.branches[
            mesmo.utils.get_index(self.branches, raise_empty_index_error=False, branch_type="line")
        ]
        self.transformers = self.branches[
            mesmo.utils.get_index(self.branches, raise_empty_index_error=False, branch_type="transformer")
        ]

        # Obtain index set for DERs.
        self.ders = pd.MultiIndex.from_frame(electric_grid_data.electric_grid_ders[["der_type", "der_name"]])

        # Obtain reference / no load voltage vector.
        self.node_voltage_vector_reference = np.zeros(len(self.nodes), dtype=complex)
        voltage_phase_factors = np.array(
            [
                np.exp(0 * 1j),  # Phase 1.
                np.exp(-2 * np.pi / 3 * 1j),  # Phase 2.
                np.exp(2 * np.pi / 3 * 1j),  # Phase 3.
            ]
        )
        for node_name, node in electric_grid_data.electric_grid_nodes.iterrows():
            # Obtain phases index & node index for positioning the node voltage in the voltage vector.
            phases_index = mesmo.utils.get_element_phases_array(node) - 1
            node_index = mesmo.utils.get_index(self.nodes, node_name=node_name)

            # Insert voltage into voltage vector.
            self.node_voltage_vector_reference[node_index] = (
                voltage_phase_factors[phases_index] * node.at["voltage"] / np.sqrt(3)
            )

        # Obtain reference / rated branch power vector.
        self.branch_power_vector_magnitude_reference = np.zeros(len(self.branches), dtype=float)
        for line_name, line in electric_grid_data.electric_grid_lines.iterrows():
            # Obtain branch index.
            branch_index = mesmo.utils.get_index(self.branches, branch_type="line", branch_name=line_name)

            # Insert rated power into branch power vector.
            self.branch_power_vector_magnitude_reference[branch_index] = (
                line.at["maximum_current"]
                * electric_grid_data.electric_grid_nodes.at[line.at["node_1_name"], "voltage"]
                / np.sqrt(3)
            )
        for transformer_name, transformer in electric_grid_data.electric_grid_transformers.iterrows():
            # Obtain branch index.
            branch_index = mesmo.utils.get_index(self.branches, branch_type="transformer", branch_name=transformer_name)

            # Insert rated power into branch flow vector.
            self.branch_power_vector_magnitude_reference[branch_index] = transformer.at["apparent_power"] / len(
                branch_index
            )  # Divide total capacity by number of phases.

        # Obtain reference / nominal DER power vector.
        self.der_power_vector_reference = (
            electric_grid_data.electric_grid_ders.loc[:, "active_power_nominal"]
            + 1.0j * electric_grid_data.electric_grid_ders.loc[:, "reactive_power_nominal"]
        ).values

        # Obtain flag for single-phase-equivalent modelling.
        if electric_grid_data.electric_grid.at["is_single_phase_equivalent"] == 1:
            if len(self.phases) != 1:
                raise ValueError(
                    f"Cannot model electric grid with {len(self.phases)} phase as single-phase-equivalent."
                )
            self.is_single_phase_equivalent = True
        else:
            self.is_single_phase_equivalent = False

        # Make modifications for single-phase-equivalent modelling.
        if self.is_single_phase_equivalent:
            self.branch_power_vector_magnitude_reference[mesmo.utils.get_index(self.branches, branch_type="line")] *= 3

        # Define sparse matrices for nodal admittance, nodal transformation,
        # branch admittance, branch incidence and der incidence matrix entries.
        self.node_admittance_matrix = sp.dok_matrix((len(self.nodes), len(self.nodes)), dtype=complex)
        self.node_transformation_matrix = sp.dok_matrix((len(self.nodes), len(self.nodes)), dtype=int)
        self.branch_admittance_1_matrix = sp.dok_matrix((len(self.branches), len(self.nodes)), dtype=complex)
        self.branch_admittance_2_matrix = sp.dok_matrix((len(self.branches), len(self.nodes)), dtype=complex)
        self.branch_incidence_1_matrix = sp.dok_matrix((len(self.branches), len(self.nodes)), dtype=int)
        self.branch_incidence_2_matrix = sp.dok_matrix((len(self.branches), len(self.nodes)), dtype=int)
        self.der_incidence_wye_matrix = sp.dok_matrix((len(self.nodes), len(self.ders)), dtype=float)
        self.der_incidence_delta_matrix = sp.dok_matrix((len(self.nodes), len(self.ders)), dtype=float)

        # Add lines to admittance, transformation and incidence matrices.
        for line_index, line in electric_grid_data.electric_grid_lines.iterrows():
            # Obtain phases vector.
            phases_vector = mesmo.utils.get_element_phases_array(line)

            # Obtain line resistance / reactance / capacitance matrix entries for the line.
            matrices_index = (
                electric_grid_data.electric_grid_line_types_matrices.loc[:, "line_type"] == line["line_type"]
            )
            resistance_matrix = electric_grid_data.electric_grid_line_types_matrices.loc[
                matrices_index, "resistance"
            ].values
            reactance_matrix = electric_grid_data.electric_grid_line_types_matrices.loc[
                matrices_index, "reactance"
            ].values
            capacitance_matrix = electric_grid_data.electric_grid_line_types_matrices.loc[
                matrices_index, "capacitance"
            ].values

            # Obtain the full line resistance and reactance matrices.
            # Data only contains upper half entries.
            matrices_full_index = np.array([[1, 2, 4], [2, 3, 5], [4, 5, 6]]) - 1
            matrices_full_index = matrices_full_index[: len(phases_vector), : len(phases_vector)]
            resistance_matrix = resistance_matrix[matrices_full_index]
            reactance_matrix = reactance_matrix[matrices_full_index]
            capacitance_matrix = capacitance_matrix[matrices_full_index]

            # Construct line series admittance matrix.
            series_admittance_matrix = np.linalg.inv((resistance_matrix + 1j * reactance_matrix) * line["length"])

            # Construct line shunt admittance.
            # Note: nF to Ω with X = 1 / (2π * f * C)
            # TODO: Check line shunt admittance.
            shunt_admittance_matrix = (
                capacitance_matrix
                * 2
                * np.pi
                * electric_grid_data.electric_grid.at["base_frequency"]
                * 1e-9
                * 0.5j
                * line["length"]
            )

            # Construct line element admittance matrices according to:
            # https://doi.org/10.1109/TPWRS.2017.2728618
            admittance_matrix_11 = series_admittance_matrix + shunt_admittance_matrix
            admittance_matrix_12 = -series_admittance_matrix
            admittance_matrix_21 = -series_admittance_matrix
            admittance_matrix_22 = series_admittance_matrix + shunt_admittance_matrix

            # Obtain indexes for positioning the line element matrices
            # in the full admittance matrices.
            node_index_1 = mesmo.utils.get_index(self.nodes, node_name=line["node_1_name"], phase=phases_vector)
            node_index_2 = mesmo.utils.get_index(self.nodes, node_name=line["node_2_name"], phase=phases_vector)
            branch_index = mesmo.utils.get_index(self.branches, branch_type="line", branch_name=line["line_name"])

            # Add line element matrices to the nodal admittance matrix.
            self.node_admittance_matrix[np.ix_(node_index_1, node_index_1)] += admittance_matrix_11
            self.node_admittance_matrix[np.ix_(node_index_1, node_index_2)] += admittance_matrix_12
            self.node_admittance_matrix[np.ix_(node_index_2, node_index_1)] += admittance_matrix_21
            self.node_admittance_matrix[np.ix_(node_index_2, node_index_2)] += admittance_matrix_22

            # Add line element matrices to the branch admittance matrices.
            self.branch_admittance_1_matrix[np.ix_(branch_index, node_index_1)] += admittance_matrix_11
            self.branch_admittance_1_matrix[np.ix_(branch_index, node_index_2)] += admittance_matrix_12
            self.branch_admittance_2_matrix[np.ix_(branch_index, node_index_1)] += admittance_matrix_21
            self.branch_admittance_2_matrix[np.ix_(branch_index, node_index_2)] += admittance_matrix_22

            # Add line element matrices to the branch incidence matrices.
            self.branch_incidence_1_matrix[np.ix_(branch_index, node_index_1)] += np.identity(
                len(branch_index), dtype=int
            )
            self.branch_incidence_2_matrix[np.ix_(branch_index, node_index_2)] += np.identity(
                len(branch_index), dtype=int
            )

        # Add transformers to admittance, transformation and incidence matrices.
        # - Note: This setup only works for transformers with exactly two windings
        #   and identical number of phases at each winding / side.

        # Define transformer factor matrices according to:
        # https://doi.org/10.1109/TPWRS.2017.2728618
        transformer_factors_1 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        transformer_factors_2 = 1 / 3 * np.array([[2, -1, -1], [-1, 2, -1], [-1, -1, 2]])
        transformer_factors_3 = 1 / np.sqrt(3) * np.array([[-1, 1, 0], [0, -1, 1], [1, 0, -1]])

        # Add transformers to admittance matrix.
        for transformer_index, transformer in electric_grid_data.electric_grid_transformers.iterrows():
            # Raise error if transformer nominal power is not valid.
            if not (transformer.at["apparent_power"] > 0):
                raise ValueError(
                    f"At transformer '{transformer.at['transformer_name']}', "
                    f"found invalid value for `apparent_power`: {transformer.at['apparent_power']}`"
                )

            # Calculate transformer admittance.
            admittance = (
                (2 * transformer.at["resistance_percentage"] / 100 + 1j * transformer.at["reactance_percentage"] / 100)
                * (
                    electric_grid_data.electric_grid_nodes.at[transformer.at["node_2_name"], "voltage"] ** 2
                    / transformer.at["apparent_power"]
                )
            ) ** -1

            # Calculate turn ratio.
            turn_ratio = (
                1.0  # TODO: Replace `1.0` with actual tap position.
                * electric_grid_data.electric_grid_nodes.at[transformer.at["node_1_name"], "voltage"]
            ) / (
                1.0  # TODO: Replace `1.0` with actual tap position.
                * electric_grid_data.electric_grid_nodes.at[transformer.at["node_2_name"], "voltage"]
            )

            # Construct transformer element admittance matrices according to:
            # https://doi.org/10.1109/TPWRS.2017.2728618
            if transformer.at["connection"] == "wye-wye":
                admittance_matrix_11 = admittance * transformer_factors_1 / turn_ratio**2
                admittance_matrix_12 = -1 * admittance * transformer_factors_1 / turn_ratio
                admittance_matrix_21 = -1 * admittance * transformer_factors_1 / turn_ratio
                admittance_matrix_22 = admittance * transformer_factors_1
            elif transformer.at["connection"] == "delta-wye":
                admittance_matrix_11 = admittance * transformer_factors_2 / turn_ratio**2
                admittance_matrix_12 = -1 * admittance * -1 * np.transpose(transformer_factors_3) / turn_ratio
                admittance_matrix_21 = -1 * admittance * -1 * transformer_factors_3 / turn_ratio
                admittance_matrix_22 = admittance * transformer_factors_1
            elif transformer.at["connection"] == "wye-delta":
                admittance_matrix_11 = admittance * transformer_factors_1 / turn_ratio**2
                admittance_matrix_12 = -1 * admittance * -1 * transformer_factors_3 / turn_ratio
                admittance_matrix_21 = -1 * admittance * -1 * np.transpose(transformer_factors_3) / turn_ratio
                admittance_matrix_22 = admittance * transformer_factors_2
            elif transformer.at["connection"] == "delta-delta":
                admittance_matrix_11 = admittance * transformer_factors_2 / turn_ratio**2
                admittance_matrix_12 = -1 * admittance * transformer_factors_2 / turn_ratio
                admittance_matrix_21 = -1 * admittance * transformer_factors_2 / turn_ratio
                admittance_matrix_22 = admittance * transformer_factors_2
            else:
                raise ValueError(f"Unknown transformer type: {transformer.at['connection']}")

            # Obtain phases vector.
            phases_vector = mesmo.utils.get_element_phases_array(transformer)

            # Obtain element admittance matrices for correct phases.
            admittance_matrix_11 = admittance_matrix_11[np.ix_(phases_vector - 1, phases_vector - 1)]
            admittance_matrix_12 = admittance_matrix_12[np.ix_(phases_vector - 1, phases_vector - 1)]
            admittance_matrix_21 = admittance_matrix_21[np.ix_(phases_vector - 1, phases_vector - 1)]
            admittance_matrix_22 = admittance_matrix_22[np.ix_(phases_vector - 1, phases_vector - 1)]

            # Obtain indexes for positioning the transformer element
            # matrices in the full matrices.
            node_index_1 = mesmo.utils.get_index(
                self.nodes, node_name=transformer.at["node_1_name"], phase=phases_vector
            )
            node_index_2 = mesmo.utils.get_index(
                self.nodes, node_name=transformer.at["node_2_name"], phase=phases_vector
            )
            branch_index = mesmo.utils.get_index(
                self.branches, branch_type="transformer", branch_name=transformer["transformer_name"]
            )

            # Add transformer element matrices to the nodal admittance matrix.
            self.node_admittance_matrix[np.ix_(node_index_1, node_index_1)] += admittance_matrix_11
            self.node_admittance_matrix[np.ix_(node_index_1, node_index_2)] += admittance_matrix_12
            self.node_admittance_matrix[np.ix_(node_index_2, node_index_1)] += admittance_matrix_21
            self.node_admittance_matrix[np.ix_(node_index_2, node_index_2)] += admittance_matrix_22

            # Add transformer element matrices to the branch admittance matrices.
            self.branch_admittance_1_matrix[np.ix_(branch_index, node_index_1)] += admittance_matrix_11
            self.branch_admittance_1_matrix[np.ix_(branch_index, node_index_2)] += admittance_matrix_12
            self.branch_admittance_2_matrix[np.ix_(branch_index, node_index_1)] += admittance_matrix_21
            self.branch_admittance_2_matrix[np.ix_(branch_index, node_index_2)] += admittance_matrix_22

            # Add transformer element matrices to the branch incidence matrices.
            self.branch_incidence_1_matrix[np.ix_(branch_index, node_index_1)] += np.identity(
                len(branch_index), dtype=int
            )
            self.branch_incidence_2_matrix[np.ix_(branch_index, node_index_2)] += np.identity(
                len(branch_index), dtype=int
            )

        # Define transformation matrix according to:
        # https://doi.org/10.1109/TPWRS.2018.2823277
        transformation_entries = np.array([[1, -1, 0], [0, 1, -1], [-1, 0, 1]])
        for node_name, node in electric_grid_data.electric_grid_nodes.iterrows():
            # Obtain node phases index.
            phases_index = mesmo.utils.get_element_phases_array(node) - 1

            # Construct node transformation matrix.
            transformation_matrix = transformation_entries[np.ix_(phases_index, phases_index)]

            # Obtain index for positioning node transformation matrix in full transformation matrix.
            node_index = mesmo.utils.get_index(self.nodes, node_name=node["node_name"])

            # Add node transformation matrix to full transformation matrix.
            self.node_transformation_matrix[np.ix_(node_index, node_index)] = transformation_matrix

        # Add DERs to DER incidence matrix.
        for der_name, der in electric_grid_data.electric_grid_ders.iterrows():
            # Obtain der connection type.
            connection = der["connection"]

            # Obtain indexes for positioning the DER in the incidence matrix.
            node_index = mesmo.utils.get_index(
                self.nodes, node_name=der["node_name"], phase=mesmo.utils.get_element_phases_array(der)
            )
            der_index = mesmo.utils.get_index(self.ders, der_name=der["der_name"])

            if connection == "wye":
                # Define incidence matrix entries.
                # - Wye ders are represented as balanced ders across all
                #   their connected phases.
                incidence_matrix = np.ones((len(node_index), 1), dtype=float) / len(node_index)
                self.der_incidence_wye_matrix[np.ix_(node_index, der_index)] = incidence_matrix

            elif connection == "delta":
                # Obtain phases of the delta der.
                phases_list = mesmo.utils.get_element_phases_array(der).tolist()

                # Select connection node based on phase arrangement of delta der.
                # TODO: Why no multi-phase delta DERs?
                # - Delta DERs must be single-phase.
                if phases_list in ([1, 2], [2, 3]):
                    node_index = [node_index[0]]
                elif phases_list == [1, 3]:
                    node_index = [node_index[1]]
                else:
                    raise ValueError(f"Unknown delta phase arrangement: {phases_list}")

                # Define incidence matrix entry.
                # - Delta ders are assumed to be single-phase.
                incidence_matrix = np.array([1])
                self.der_incidence_delta_matrix[np.ix_(node_index, der_index)] = incidence_matrix

            else:
                raise ValueError(f"Unknown der connection type: {connection}")

        # Make modifications for single-phase-equivalent modelling.
        if self.is_single_phase_equivalent:
            self.der_incidence_wye_matrix /= 3
            # Note that there won't be any delta loads in the single-phase-equivalent grid.

        # Convert sparse matrices for nodal admittance, nodal transformation,
        # branch admittance, branch incidence and der incidence matrices.
        # - Converting from DOK to CSR format for more efficient calculations
        #   according to <https://docs.scipy.org/doc/scipy/reference/sparse.html>.
        self.node_admittance_matrix = self.node_admittance_matrix.tocsr()
        self.node_transformation_matrix = self.node_transformation_matrix.tocsr()
        self.branch_admittance_1_matrix = self.branch_admittance_1_matrix.tocsr()
        self.branch_admittance_2_matrix = self.branch_admittance_2_matrix.tocsr()
        self.branch_incidence_1_matrix = self.branch_incidence_1_matrix.tocsr()
        self.branch_incidence_2_matrix = self.branch_incidence_2_matrix.tocsr()
        self.der_incidence_wye_matrix = self.der_incidence_wye_matrix.tocsr()
        self.der_incidence_delta_matrix = self.der_incidence_delta_matrix.tocsr()

        # Define shorthands for no-source variables.
        # TODO: Add in class documentation.
        # TODO: Replace local variables in power flow / linear models.
        self.node_admittance_matrix_no_source = self.node_admittance_matrix[
            np.ix_(
                mesmo.utils.get_index(self.nodes, node_type="no_source"),
                mesmo.utils.get_index(self.nodes, node_type="no_source"),
            )
        ]
        self.node_admittance_matrix_source_to_no_source = self.node_admittance_matrix[
            np.ix_(
                mesmo.utils.get_index(self.nodes, node_type="no_source"),
                mesmo.utils.get_index(self.nodes, node_type="source"),
            )
        ]
        self.node_transformation_matrix_no_source = self.node_transformation_matrix[
            np.ix_(
                mesmo.utils.get_index(self.nodes, node_type="no_source"),
                mesmo.utils.get_index(self.nodes, node_type="no_source"),
            )
        ]
        self.der_incidence_wye_matrix_no_source = self.der_incidence_wye_matrix[
            np.ix_(mesmo.utils.get_index(self.nodes, node_type="no_source"), range(len(self.ders)))
        ]
        self.der_incidence_delta_matrix_no_source = self.der_incidence_delta_matrix[
            np.ix_(mesmo.utils.get_index(self.nodes, node_type="no_source"), range(len(self.ders)))
        ]
        self.node_voltage_vector_reference_no_source = self.node_voltage_vector_reference[
            mesmo.utils.get_index(self.nodes, node_type="no_source")
        ]
        self.node_voltage_vector_reference_source = self.node_voltage_vector_reference[
            mesmo.utils.get_index(self.nodes, node_type="source")
        ]

        # Calculate inverse of no-source node admittance matrix.
        # - Raise error if not invertible.
        # - Only checking invertibility of no-source node admittance matrix, because full node admittance matrix may
        #   be non-invertible, e.g. zero entries when connecting a multi-phase line at three-phase source node.
        try:
            self.node_admittance_matrix_no_source_inverse = scipy.sparse.linalg.inv(
                self.node_admittance_matrix_no_source.tocsc()
            )
            assert not np.isnan(self.node_admittance_matrix_no_source_inverse.data).any()
        except (RuntimeError, AssertionError) as exception:
            raise (
                ValueError(f"Node admittance matrix could not be inverted. Please check electric grid definition.")
            ) from exception

    @staticmethod
    def process_line_types_overhead(
        electric_grid_data: mesmo.data_interface.ElectricGridData,
    ) -> mesmo.data_interface.ElectricGridData:
        """Process overhead line type definitions in electric grid data object."""

        # Process over-head line type definitions.
        for line_type, line_type_data in electric_grid_data.electric_grid_line_types_overhead.iterrows():

            # Obtain data shorthands.
            # - Only for phases which have `conductor_id` defined in `electric_grid_line_types_overhead`.
            phases = pd.Index(
                [
                    1 if pd.notnull(line_type_data.at["phase_1_conductor_id"]) else None,
                    2 if pd.notnull(line_type_data.at["phase_2_conductor_id"]) else None,
                    3 if pd.notnull(line_type_data.at["phase_3_conductor_id"]) else None,
                    "n" if pd.notnull(line_type_data.at["neutral_conductor_id"]) else None,
                ]
            ).dropna()
            phase_conductor_id = pd.Series(
                {
                    1: line_type_data.at["phase_1_conductor_id"],
                    2: line_type_data.at["phase_2_conductor_id"],
                    3: line_type_data.at["phase_3_conductor_id"],
                    "n": line_type_data.at["neutral_conductor_id"],
                }
            ).loc[phases]
            phase_y = pd.Series(
                {
                    1: line_type_data.at["phase_1_y"],
                    2: line_type_data.at["phase_2_y"],
                    3: line_type_data.at["phase_3_y"],
                    "n": line_type_data.at["neutral_y"],
                }
            ).loc[phases]
            phase_xy = pd.Series(
                {
                    1: np.array([line_type_data.at["phase_1_x"], line_type_data.at["phase_1_y"]]),
                    2: np.array([line_type_data.at["phase_2_x"], line_type_data.at["phase_2_y"]]),
                    3: np.array([line_type_data.at["phase_3_x"], line_type_data.at["phase_3_y"]]),
                    "n": np.array([line_type_data.at["neutral_x"], line_type_data.at["neutral_y"]]),
                }
            ).loc[phases]
            phase_conductor_diameter = (
                pd.Series(
                    [
                        electric_grid_data.electric_grid_line_types_overhead_conductors.at[
                            phase_conductor_id.at[phase], "conductor_diameter"
                        ]
                        for phase in phases
                    ],
                    index=phases,
                )
                * 1e-3  # mm to m.
            )
            phase_conductor_geometric_mean_radius = (
                pd.Series(
                    [
                        electric_grid_data.electric_grid_line_types_overhead_conductors.at[
                            phase_conductor_id.at[phase], "conductor_geometric_mean_radius"
                        ]
                        for phase in phases
                    ],
                    index=phases,
                )
                * 1e-3  # mm to m.
            )
            phase_conductor_resistance = pd.Series(
                [
                    electric_grid_data.electric_grid_line_types_overhead_conductors.at[
                        phase_conductor_id.at[phase], "conductor_resistance"
                    ]
                    for phase in phases
                ],
                index=phases,
            )
            phase_conductor_maximum_current = pd.Series(
                [
                    electric_grid_data.electric_grid_line_types_overhead_conductors.at[
                        phase_conductor_id.at[phase], "conductor_maximum_current"
                    ]
                    for phase in phases
                ],
                index=phases,
            )

            # Obtain shorthands for neutral / non-neutral phases.
            # - This is needed for Kron reduction.
            phases_neutral = phases[phases.isin(["n"])]
            phases_non_neutral = phases[~phases.isin(["n"])]

            # Other parameter shorthands.
            frequency = electric_grid_data.electric_grid.at["base_frequency"]  # In Hz.
            earth_resistivity = line_type_data.at["earth_resistivity"]  # In Ωm.
            air_permittivity = line_type_data.at["air_permittivity"]  # In nF/km.
            g_factor = 1e-4  # In Ω/km from 0.1609347e-3 Ω/mile from Kersting <https://doi.org/10.1201/9781315120782>.

            # Obtain impedance matrix in Ω/km based on Kersting <https://doi.org/10.1201/9781315120782>.
            z_matrix = pd.DataFrame(index=phases, columns=phases, dtype=complex)
            for phase_row, phase_col in itertools.product(phases, phases):
                # Calculate geometric parameters.
                d_distance = np.linalg.norm(phase_xy.at[phase_row] - phase_xy.at[phase_col])
                s_distance = np.linalg.norm(phase_xy.at[phase_row] - np.array([1, -1]) * phase_xy.at[phase_col])
                s_angle = np.pi / 2 - np.arcsin((phase_y.at[phase_row] + phase_y.at[phase_col]) / s_distance)
                # Calculate Kersting / Carson parameters.
                k_factor = 8.565e-4 * s_distance * np.sqrt(frequency / earth_resistivity)
                p_factor = (
                    np.pi / 8
                    - (3 * np.sqrt(2)) ** -1 * k_factor * np.cos(s_angle)
                    - k_factor**2 / 16 * np.cos(2 * s_angle) * (0.6728 + np.log(2 / k_factor))
                )
                q_factor = (
                    -0.0386 + 0.5 * np.log(2 / k_factor) + (3 * np.sqrt(2)) ** -1 * k_factor * np.cos(2 * s_angle)
                )
                x_factor = (
                    2
                    * np.pi
                    * frequency
                    * g_factor
                    * np.log(phase_conductor_diameter[phase_row] / phase_conductor_geometric_mean_radius.at[phase_row])
                )
                # Calculate admittance according to Kersting / Carson <https://doi.org/10.1201/9781315120782>.
                if phase_row == phase_col:
                    z_matrix.at[phase_row, phase_col] = (
                        phase_conductor_resistance.at[phase_row]
                        + 4 * np.pi * frequency * p_factor * g_factor
                        + 1j
                        * (
                            x_factor
                            + 2
                            * np.pi
                            * frequency
                            * g_factor
                            * np.log(s_distance / phase_conductor_diameter[phase_row])
                            + 4 * np.pi * frequency * q_factor * g_factor
                        )
                    )
                else:
                    z_matrix.at[phase_row, phase_col] = 4 * np.pi * frequency * p_factor * g_factor + 1j * (
                        2 * np.pi * frequency * g_factor * np.log(s_distance / d_distance)
                        + 4 * np.pi * frequency * q_factor * g_factor
                    )

            # Apply Kron reduction.
            z_matrix = pd.DataFrame(
                (
                    z_matrix.loc[phases_non_neutral, phases_non_neutral].values
                    - z_matrix.loc[phases_non_neutral, phases_neutral].values
                    @ z_matrix.loc[phases_neutral, phases_neutral].values ** -1  # Inverse of scalar value.
                    @ z_matrix.loc[phases_neutral, phases_non_neutral].values
                ),
                index=phases_non_neutral,
                columns=phases_non_neutral,
            )

            # Obtain potentials matrix in km/nF based on Kersting <https://doi.org/10.1201/9781315120782>.
            p_matrix = pd.DataFrame(index=phases, columns=phases, dtype=float)
            for phase_row, phase_col in itertools.product(phases, phases):
                # Calculate geometric parameters.
                d_distance = np.linalg.norm(phase_xy.at[phase_row] - phase_xy.at[phase_col])
                s_distance = np.linalg.norm(phase_xy.at[phase_row] - np.array([1, -1]) * phase_xy.at[phase_col])
                # Calculate potential according to Kersting <https://doi.org/10.1201/9781315120782>.
                if phase_row == phase_col:
                    p_matrix.at[phase_row, phase_col] = (
                        1 / (2 * np.pi * air_permittivity) * np.log(s_distance / phase_conductor_diameter.at[phase_row])
                    )
                else:
                    p_matrix.at[phase_row, phase_col] = (
                        1 / (2 * np.pi * air_permittivity) * np.log(s_distance / d_distance)
                    )

            # Apply Kron reduction.
            p_matrix = pd.DataFrame(
                (
                    p_matrix.loc[phases_non_neutral, phases_non_neutral].values
                    - p_matrix.loc[phases_non_neutral, phases_neutral].values
                    @ p_matrix.loc[phases_neutral, phases_neutral].values ** -1  # Inverse of scalar value.
                    @ p_matrix.loc[phases_neutral, phases_non_neutral].values
                ),
                index=phases_non_neutral,
                columns=phases_non_neutral,
            )

            # Obtain capacitance matrix in nF/km.
            c_matrix = pd.DataFrame(np.linalg.inv(p_matrix), index=phases_non_neutral, columns=phases_non_neutral)

            # Obtain final element matrices.
            resistance_matrix = z_matrix.apply(np.real)  # In Ω/km.
            reactance_matrix = z_matrix.apply(np.imag)  # In Ω/km.
            capacitance_matrix = c_matrix  # In nF/km.

            # Add to line type matrices definition.
            for phase_row in phases_non_neutral:
                for phase_col in phases_non_neutral[phases_non_neutral <= phase_row]:
                    electric_grid_data.electric_grid_line_types_matrices = (
                        electric_grid_data.electric_grid_line_types_matrices.append(
                            pd.Series(
                                {
                                    "line_type": line_type,
                                    "row": phase_row,
                                    "col": phase_col,
                                    "resistance": resistance_matrix.at[phase_row, phase_col],
                                    "reactance": reactance_matrix.at[phase_row, phase_col],
                                    "capacitance": capacitance_matrix.at[phase_row, phase_col],
                                }
                            ),
                            ignore_index=True,
                        )
                    )

            # Obtain number of phases.
            electric_grid_data.electric_grid_line_types.loc[line_type, "n_phases"] = len(phases_non_neutral)

            # Obtain maximum current.
            # TODO: Validate this.
            electric_grid_data.electric_grid_line_types.loc[
                line_type, "maximum_current"
            ] = phase_conductor_maximum_current.loc[phases_non_neutral].mean()

        return electric_grid_data


class ElectricGridModelDefault(ElectricGridModel):
    """`ElectricGridModelDefault` is a placeholder for `ElectricGridModel` for backwards compatibility and will
    be removed in a future version of MESMO.
    """

    def __init__(self, *args, **kwargs):

        # Issue warning when using this class.
        logger.warning(
            "`ElectricGridModelDefault` is a placeholder for `ElectricGridModel` for backwards compatibility and will "
            "be removed in a future version of MESMO."
        )

        super().__init__(*args, **kwargs)


class ElectricGridModelOpenDSS(ElectricGridModel):
    """OpenDSS electric grid model object.

    - Instantiate OpenDSS circuit by running generating OpenDSS commands corresponding to given `electric_grid_data`,
      utilizing the `OpenDSSDirect.py` package.
    - The OpenDSS circuit can be accessed with the API of
      `OpenDSSDirect.py`: http://dss-extensions.org/OpenDSSDirect.py/opendssdirect.html
    - Due to dependency on `OpenDSSDirect.py`, creating multiple objects of this type may result in erroneous behavior.

    :syntax:
        - ``ElectricGridModelOpenDSS(electric_grid_data)``: Initialize OpenDSS circuit model for given
          `electric_grid_data`.
        - ``ElectricGridModelOpenDSS(scenario_name)`` Initialize OpenDSS circuit model for given `scenario_name`.
          The required `electric_grid_data` is obtained from the database.

    Parameters:
        scenario_name (str): MESMO scenario name.
        electric_grid_data (mesmo.data_interface.ElectricGridData): Electric grid data object.

    Attributes:
        phases (pd.Index): Index set of the phases.
        node_names (pd.Index): Index set of the node names.
        node_types (pd.Index): Index set of the node types.
        line_names (pd.Index): Index set of the line names.
        transformer_names (pd.Index): Index set of the transformer names.
        branch_names (pd.Index): Index set of the branch names, i.e., all line names and transformer names.
        branch_types (pd.Index): Index set of the branch types.
        der_names (pd.Index): Index set of the DER names.
        der_types (pd.Index): Index set of the DER types.
        nodes (pd.Index): Multi-level / tuple index set of the node types, node names and phases
            corresponding to the dimension of the node admittance matrices.
        branches (pd.Index): Multi-level / tuple index set of the branch types, branch names and phases
            corresponding to the dimension of the branch admittance matrices.
        lines (pd.Index): Multi-level / tuple index set of the branch types, branch names and phases
            for the lines only.
        transformers (pd.Index): Multi-level / tuple index set of the branch types, branch names and phases
            for the transformers only.
        ders (pd.Index): Index set of the DER names, corresponding to the dimension of the DER power vector.
        node_voltage_vector_reference (np.ndarray): Node voltage reference / no load vector.
        branch_power_vector_magnitude_reference (np.ndarray): Branch power reference / rated power vector.
        der_power_vector_reference (np.ndarray): DER power reference / nominal power vector.
        is_single_phase_equivalent (bool): Singe-phase-equivalent modelling flag. If true, electric grid is modelled
            as single-phase-equivalent of three-phase balanced system.
        circuit_name (str): Circuit name, stored for validation that the correct OpenDSS model is being accessed.
        electric_grid_data: (mesmo.data_interface.ElectricGridData): Electric grid data object, stored for
            possible reinitialization of the OpenDSS model.
    """

    circuit_name: str
    electric_grid_data: mesmo.data_interface.ElectricGridData

    @multimethod.multimethod
    def __init__(self, scenario_name: str):

        # Obtain electric grid data.
        electric_grid_data = mesmo.data_interface.ElectricGridData(scenario_name)

        self.__init__(electric_grid_data)

    @multimethod.multimethod
    def __init__(self, electric_grid_data: mesmo.data_interface.ElectricGridData):

        # TODO: Add reset method to ensure correct circuit model is set in OpenDSS when handling multiple models.

        # Obtain electric grid indexes, via `ElectricGridModel.__init__()`.
        super().__init__(electric_grid_data)

        # Obtain circuit name.
        self.circuit_name = electric_grid_data.electric_grid.at["electric_grid_name"]

        # Store electric grid data.
        self.electric_grid_data = electric_grid_data

        # Clear OpenDSS.
        opendss_command_string = "clear"
        logger.debug(f"opendss_command_string = \n{opendss_command_string}")
        opendssdirect.run_command(opendss_command_string)

        # Obtain source voltage.
        source_voltage = electric_grid_data.electric_grid_nodes.at[
            electric_grid_data.electric_grid.at["source_node_name"], "voltage"
        ]

        # Adjust source voltage for single-phase, non-single-phase-equivalent modelling.
        if (len(self.phases) == 1) and not self.is_single_phase_equivalent:
            source_voltage /= np.sqrt(3)

        # Add circuit info to OpenDSS command string.
        opendss_command_string = (
            f"set defaultbasefrequency={electric_grid_data.electric_grid.at['base_frequency']}"
            + f"\nnew circuit.{self.circuit_name}"
            + f" phases={len(self.phases)}"
            + f" bus1={electric_grid_data.electric_grid.at['source_node_name']}"
            + f" basekv={source_voltage / 1000}"
            + f" mvasc3=9999999999 9999999999"  # Set near-infinite power limit for source node.
        )

        # Create circuit in OpenDSS.
        logger.debug(f"opendss_command_string = \n{opendss_command_string}")
        opendssdirect.run_command(opendss_command_string)

        # Define line codes.
        for line_type_index, line_type in electric_grid_data.electric_grid_line_types.iterrows():
            # Obtain line resistance and reactance matrix entries for the line.
            matrices = electric_grid_data.electric_grid_line_types_matrices.loc[
                (electric_grid_data.electric_grid_line_types_matrices.loc[:, "line_type"] == line_type.at["line_type"]),
                ["resistance", "reactance", "capacitance"],
            ]

            # Obtain number of phases.
            # - Only define as line types for as many phases as needed for current grid.
            n_phases = min(line_type.at["n_phases"], len(self.phases))

            # Add line type name and number of phases to OpenDSS command string.
            opendss_command_string = f"new linecode.{line_type.at['line_type']}" + f" nphases={n_phases}"

            # Add resistance and reactance matrix entries to OpenDSS command string,
            # with formatting depending on number of phases.
            if n_phases == 1:
                opendss_command_string += (
                    " rmatrix = "
                    + "[{:.8f}]".format(*matrices.loc[:, "resistance"])
                    + " xmatrix = "
                    + "[{:.8f}]".format(*matrices.loc[:, "reactance"])
                    + " cmatrix = "
                    + "[{:.8f}]".format(*matrices.loc[:, "capacitance"])
                )
            elif n_phases == 2:
                opendss_command_string += (
                    " rmatrix = "
                    + "[{:.8f} | {:.8f} {:.8f}]".format(*matrices.loc[:, "resistance"])
                    + " xmatrix = "
                    + "[{:.8f} | {:.8f} {:.8f}]".format(*matrices.loc[:, "reactance"])
                    + " cmatrix = "
                    + "[{:.8f} | {:.8f} {:.8f}]".format(*matrices.loc[:, "capacitance"])
                )
            elif n_phases == 3:
                opendss_command_string += (
                    " rmatrix = "
                    + "[{:.8f} | {:.8f} {:.8f} | {:.8f} {:.8f} {:.8f}]".format(*matrices.loc[:, "resistance"])
                    + f" xmatrix = "
                    + "[{:.8f} | {:.8f} {:.8f} | {:.8f} {:.8f} {:.8f}]".format(*matrices.loc[:, "reactance"])
                    + f" cmatrix = "
                    + "[{:.8f} | {:.8f} {:.8f} | {:.8f} {:.8f} {:.8f}]".format(*matrices.loc[:, "capacitance"])
                )

            # Create line code in OpenDSS.
            logger.debug(f"opendss_command_string = \n{opendss_command_string}")
            opendssdirect.run_command(opendss_command_string)

        # Define lines.
        for line_index, line in electric_grid_data.electric_grid_lines.iterrows():
            # Obtain number of phases for the line.
            n_phases = len(mesmo.utils.get_element_phases_array(line))

            # Add line name, phases, node connections, line type and length
            # to OpenDSS command string.
            opendss_command_string = (
                f"new line.{line['line_name']}"
                + f" phases={n_phases}"
                + f" bus1={line['node_1_name']}{mesmo.utils.get_element_phases_string(line)}"
                + f" bus2={line['node_2_name']}{mesmo.utils.get_element_phases_string(line)}"
                + f" linecode={line['line_type']}"
                + f" length={line['length']}"
            )

            # Create line in OpenDSS.
            logger.debug(f"opendss_command_string = \n{opendss_command_string}")
            opendssdirect.run_command(opendss_command_string)

        # Define transformers.
        for transformer_index, transformer in electric_grid_data.electric_grid_transformers.iterrows():
            # Obtain number of phases.
            n_phases = len(mesmo.utils.get_element_phases_array(transformer))

            # Add transformer name, number of phases / windings and reactances to OpenDSS command string.
            opendss_command_string = (
                f"new transformer.{transformer.at['transformer_name']}"
                + f" phases={n_phases}"
                + f" windings=2"
                + f" xscarray=[{transformer.at['reactance_percentage']}]"
            )

            # Add windings to OpenDSS command string.
            windings = [1, 2]
            for winding in windings:
                # Obtain nominal voltage level for each winding.
                voltage = electric_grid_data.electric_grid_nodes.at[transformer.at[f"node_{winding}_name"], "voltage"]

                # Obtain node phases connection string for each winding.
                connection = transformer.at["connection"].split("-")[winding - 1]
                if connection == "wye":
                    node_phases_string = (
                        mesmo.utils.get_element_phases_string(transformer) + ".0"  # Enforce wye-grounded connection.
                    )
                elif connection == "delta":
                    node_phases_string = mesmo.utils.get_element_phases_string(transformer)
                else:
                    raise ValueError(f"Unknown transformer connection type: {connection}")

                # Add node connection, nominal voltage / power, resistance and maximum / minimum tap level
                # to OpenDSS command string for each winding.
                opendss_command_string += (
                    f" wdg={winding}"
                    + f" bus={transformer.at[f'node_{winding}_name']}"
                    + node_phases_string
                    + f" conn={connection}"
                    + f" kv={voltage / 1000}"
                    + f" kva={transformer.at['apparent_power'] / 1000}"
                    + f" %r={transformer.at['resistance_percentage']}"
                    + f" maxtap="
                    + f"{transformer.at['tap_maximum_voltage_per_unit']}"
                    + f" mintap="
                    + f"{transformer.at['tap_minimum_voltage_per_unit']}"
                )

            # Create transformer in OpenDSS.
            logger.debug(f"opendss_command_string = \n{opendss_command_string}")
            opendssdirect.run_command(opendss_command_string)

        # Define DERs.
        # TODO: At the moment, all DERs are modelled as loads in OpenDSS.
        for der_index, der in electric_grid_data.electric_grid_ders.iterrows():
            # Obtain number of phases for the DER.
            n_phases = len(mesmo.utils.get_element_phases_array(der))

            # Obtain nominal voltage level for the DER.
            voltage = electric_grid_data.electric_grid_nodes.at[der["node_name"], "voltage"]
            # Convert to line-to-neutral voltage for single-phase DERs, according to:
            # https://sourceforge.net/p/electricdss/discussion/861976/thread/9c9e0efb/
            # - Not needed for single-phase-equivalent modelling.
            if (n_phases == 1) and not self.is_single_phase_equivalent:
                voltage /= np.sqrt(3)

            # Add explicit ground-phase connection for single-phase, wye DERs, according to:
            # https://sourceforge.net/p/electricdss/discussion/861976/thread/d420e8fb/
            # - This does not seem to make a difference if omitted, but is kept here to follow the recommendation.
            # - Not needed for single-phase-equivalent modelling.
            if (n_phases == 1) and (der["connection"] == "wye") and not self.is_single_phase_equivalent:
                ground_phase_string = ".0"
            else:
                ground_phase_string = ""

            # Add node connection, model type, voltage, nominal power to OpenDSS command string.
            opendss_command_string = (
                f"new load.{der['der_name']}"
                + f" bus1={der['node_name']}{ground_phase_string}{mesmo.utils.get_element_phases_string(der)}"
                + f" phases={n_phases}"
                + f" conn={der['connection']}"
                # All loads are modelled as constant P/Q according to:
                # OpenDSS Manual April 2018, page 150, "Model"
                + f" model=1"
                + f" kv={voltage / 1000}"
                + f" kw={- der['active_power_nominal'] / 1000}"
                + f" kvar={- der['reactive_power_nominal'] / 1000}"
                # Set low V_min to avoid switching to impedance model according to:
                # OpenDSS Manual April 2018, page 150, "Vminpu"
                + f" vminpu=0.6"
                # Set high V_max to avoid switching to impedance model according to:
                # OpenDSS Manual April 2018, page 150, "Vmaxpu"
                + f" vmaxpu=1.4"
            )

            # Create DER in OpenDSS.
            logger.debug(f"opendss_command_string = \n{opendss_command_string}")
            opendssdirect.run_command(opendss_command_string)

        # Obtain voltage bases.
        voltage_bases = np.unique(electric_grid_data.electric_grid_nodes.loc[:, "voltage"].values / 1000).tolist()

        # Set control mode and voltage bases.
        opendss_command_string = f"set voltagebases={voltage_bases}" + f"\nset controlmode=off" + f"\ncalcvoltagebases"
        logger.debug(f"opendss_command_string = \n{opendss_command_string}")
        opendssdirect.run_command(opendss_command_string)

        # Set solution mode to "single snapshot power flow" according to:
        # OpenDSSComDoc, November 2016, page 1
        opendss_command_string = "set mode=0"
        logger.debug(f"opendss_command_string = \n{opendss_command_string}")
        opendssdirect.run_command(opendss_command_string)


class ElectricGridDEROperationResults(mesmo.utils.ResultsBase):

    der_active_power_vector: pd.DataFrame
    der_active_power_vector_per_unit: pd.DataFrame
    der_reactive_power_vector: pd.DataFrame
    der_reactive_power_vector_per_unit: pd.DataFrame


class ElectricGridOperationResults(ElectricGridDEROperationResults):

    electric_grid_model: ElectricGridModel
    node_voltage_magnitude_vector: pd.DataFrame
    node_voltage_magnitude_vector_per_unit: pd.DataFrame
    node_voltage_angle_vector: pd.DataFrame
    branch_power_magnitude_vector_1: pd.DataFrame
    branch_power_magnitude_vector_1_per_unit: pd.DataFrame
    branch_active_power_vector_1: pd.DataFrame
    branch_active_power_vector_1_per_unit: pd.DataFrame
    branch_reactive_power_vector_1: pd.DataFrame
    branch_reactive_power_vector_1_per_unit: pd.DataFrame
    branch_power_magnitude_vector_2: pd.DataFrame
    branch_power_magnitude_vector_2_per_unit: pd.DataFrame
    branch_active_power_vector_2: pd.DataFrame
    branch_active_power_vector_2_per_unit: pd.DataFrame
    branch_reactive_power_vector_2: pd.DataFrame
    branch_reactive_power_vector_2_per_unit: pd.DataFrame
    loss_active: pd.DataFrame
    loss_reactive: pd.DataFrame


class ElectricGridDLMPResults(mesmo.utils.ResultsBase):

    electric_grid_energy_dlmp_node_active_power: pd.DataFrame
    electric_grid_voltage_dlmp_node_active_power: pd.DataFrame
    electric_grid_congestion_dlmp_node_active_power: pd.DataFrame
    electric_grid_loss_dlmp_node_active_power: pd.DataFrame
    electric_grid_total_dlmp_node_active_power: pd.DataFrame
    electric_grid_voltage_dlmp_node_reactive_power: pd.DataFrame
    electric_grid_congestion_dlmp_node_reactive_power: pd.DataFrame
    electric_grid_loss_dlmp_node_reactive_power: pd.DataFrame
    electric_grid_energy_dlmp_node_reactive_power: pd.DataFrame
    electric_grid_total_dlmp_node_reactive_power: pd.DataFrame
    electric_grid_energy_dlmp_der_active_power: pd.DataFrame
    electric_grid_voltage_dlmp_der_active_power: pd.DataFrame
    electric_grid_congestion_dlmp_der_active_power: pd.DataFrame
    electric_grid_loss_dlmp_der_active_power: pd.DataFrame
    electric_grid_total_dlmp_der_active_power: pd.DataFrame
    electric_grid_voltage_dlmp_der_reactive_power: pd.DataFrame
    electric_grid_congestion_dlmp_der_reactive_power: pd.DataFrame
    electric_grid_loss_dlmp_der_reactive_power: pd.DataFrame
    electric_grid_energy_dlmp_der_reactive_power: pd.DataFrame
    electric_grid_total_dlmp_der_reactive_power: pd.DataFrame
    electric_grid_total_dlmp_price_timeseries: pd.DataFrame


class PowerFlowSolutionBase(mesmo.utils.ObjectBase):
    """Power flow solution base object consisting of DER power vector and the corresponding solution for
    nodal voltage vector / branch power vector and total loss, all complex valued.
    """

    der_power_vector: np.ndarray
    node_voltage_vector: np.ndarray
    branch_power_vector_1: np.ndarray
    branch_power_vector_2: np.ndarray
    loss: complex


class PowerFlowSolutionFixedPoint(PowerFlowSolutionBase):
    """Fixed point power flow solution object."""

    @multimethod.multimethod
    def __init__(self, scenario_name: str, **kwargs):

        # Obtain `electric_grid_model`.
        electric_grid_model = ElectricGridModel(scenario_name)

        self.__init__(electric_grid_model, **kwargs)

    @multimethod.multimethod
    def __init__(self, electric_grid_model: ElectricGridModel, **kwargs):

        # Obtain `der_power_vector`, assuming nominal power conditions.
        der_power_vector = electric_grid_model.der_power_vector_reference

        self.__init__(electric_grid_model, der_power_vector, **kwargs)

    @multimethod.multimethod
    def __init__(self, electric_grid_model: ElectricGridModel, der_power_vector: np.ndarray, **kwargs):

        # Store DER power vector.
        self.der_power_vector = der_power_vector.ravel()

        # Obtain voltage solution.
        self.node_voltage_vector = self.get_voltage(electric_grid_model, self.der_power_vector, **kwargs)

        # Obtain branch flow solution.
        (self.branch_power_vector_1, self.branch_power_vector_2) = self.get_branch_power(
            electric_grid_model, self.node_voltage_vector
        )

        # Obtain loss solution.
        self.loss = self.get_loss(electric_grid_model, self.node_voltage_vector)

    @staticmethod
    def check_solution_conditions(
        electric_grid_model: ElectricGridModel,
        node_power_vector_wye_initial_no_source: np.ndarray,
        node_power_vector_delta_initial_no_source: np.ndarray,
        node_power_vector_wye_candidate_no_source: np.ndarray,
        node_power_vector_delta_candidate_no_source: np.ndarray,
        node_voltage_vector_initial_no_source: np.ndarray,
    ) -> bool:
        """Check conditions for fixed-point solution existence, uniqueness and non-singularity for
         given power vector candidate and initial point.

        - Conditions are formulated according to: <https://arxiv.org/pdf/1702.03310.pdf>
        - Note the performance issues of this condition check algorithm due to the
          requirement for matrix inversions / solving of linear equations.
        """

        # Calculate norm of the initial nodal power vector.
        xi_initial = np.max(
            np.sum(
                np.abs(
                    (electric_grid_model.node_voltage_vector_reference_no_source**-1)
                    * scipy.sparse.linalg.spsolve(
                        electric_grid_model.node_admittance_matrix_no_source,
                        (
                            (electric_grid_model.node_voltage_vector_reference_no_source**-1)
                            * node_power_vector_wye_initial_no_source
                        ),
                    )
                ),
                axis=1,
            )
        ) + np.max(
            np.sum(
                np.abs(
                    (electric_grid_model.node_voltage_vector_reference_no_source**-1)
                    * scipy.sparse.linalg.spsolve(
                        electric_grid_model.node_admittance_matrix_no_source,
                        (
                            (
                                electric_grid_model.node_transformation_matrix_no_source
                                * (
                                    np.abs(electric_grid_model.node_transformation_matrix_no_source)
                                    @ np.abs(electric_grid_model.node_voltage_vector_reference_no_source)
                                )
                                ** -1
                            )
                            * node_power_vector_delta_initial_no_source
                        ),
                    )
                ),
                axis=1,
            )
        )

        # Calculate norm of the candidate nodal power vector.
        xi_candidate = np.max(
            np.sum(
                np.abs(
                    (electric_grid_model.node_voltage_vector_reference_no_source**-1)
                    * scipy.sparse.linalg.spsolve(
                        electric_grid_model.node_admittance_matrix_no_source,
                        (
                            (electric_grid_model.node_voltage_vector_reference_no_source**-1)
                            * (node_power_vector_wye_candidate_no_source - node_power_vector_wye_initial_no_source)
                        ),
                    )
                ),
                axis=1,
            )
        ) + np.max(
            np.sum(
                np.abs(
                    (electric_grid_model.node_voltage_vector_reference_no_source**-1)
                    * scipy.sparse.linalg.spsolve(
                        electric_grid_model.node_admittance_matrix_no_source,
                        (
                            (
                                electric_grid_model.node_transformation_matrix_no_source
                                * (
                                    np.abs(electric_grid_model.node_transformation_matrix_no_source)
                                    @ np.abs(electric_grid_model.node_voltage_vector_reference_no_source)
                                )
                                ** -1
                            )
                            * (node_power_vector_delta_candidate_no_source - node_power_vector_delta_initial_no_source)
                        ),
                    )
                ),
                axis=1,
            )
        )

        # Calculate norm of the initial nodal voltage vector.
        gamma = np.min(
            [
                np.min(
                    np.abs(node_voltage_vector_initial_no_source)
                    / np.abs(electric_grid_model.node_voltage_vector_reference_no_source)
                ),
                np.min(
                    np.abs(
                        electric_grid_model.node_transformation_matrix_no_source * node_voltage_vector_initial_no_source
                    )
                    / (
                        np.abs(electric_grid_model.node_transformation_matrix_no_source)
                        * np.abs(electric_grid_model.node_voltage_vector_reference_no_source)
                    )
                ),
            ]
        )

        # Obtain conditions for solution existence, uniqueness and non-singularity.
        condition_initial = xi_initial < (gamma**2)
        condition_candidate = xi_candidate < (0.25 * (((gamma**2) - xi_initial) / gamma) ** 2)
        is_valid = condition_initial & condition_candidate

        # If `condition_initial` is violated, the given initial nodal voltage vector  and power vectors are not valid.
        # This suggests an error in the problem setup and hence triggers a warning.
        if ~condition_initial:
            logger.warning("Fixed point solution condition is not satisfied for the provided initial point.")

        return is_valid

    @staticmethod
    def get_voltage(
        electric_grid_model: ElectricGridModel,
        der_power_vector: np.ndarray,
        outer_iteration_limit=100,
        outer_solution_algorithm="check_solution",  # Choices: `check_conditions`, `check_solution`.
        power_candidate_iteration_limit=100,
        power_candidate_reduction_factor=0.5,
        voltage_iteration_limit=100,
        voltage_tolerance=1e-2,
    ) -> np.ndarray:
        """Get nodal voltage vector by solving with the fixed point algorithm.

        - Initial DER power vector / node voltage vector must be a valid
          solution to te fixed-point equation, e.g., a previous solution from a past
          operation point.
        - Fixed point equation according to: <https://arxiv.org/pdf/1702.03310.pdf>
        """

        # TODO: Add proper documentation.
        # TODO: Validate fixed-point solution conditions.

        # Debug message.
        logger.debug("Starting fixed point solution algorithm...")

        # Obtain nodal power vectors.
        node_power_vector_wye_no_source = (
            electric_grid_model.der_incidence_wye_matrix_no_source @ np.transpose([der_power_vector])
        ).ravel()
        node_power_vector_delta_no_source = (
            electric_grid_model.der_incidence_delta_matrix_no_source @ np.transpose([der_power_vector])
        ).ravel()

        # Obtain initial nodal power and voltage vectors, assuming no power conditions.
        # TODO: Enable passing previous solution for fixed-point initialization.
        node_power_vector_wye_initial_no_source = np.zeros(node_power_vector_wye_no_source.shape, dtype=complex)
        node_power_vector_delta_initial_no_source = np.zeros(node_power_vector_delta_no_source.shape, dtype=complex)
        node_voltage_vector_initial_no_source = electric_grid_model.node_voltage_vector_reference_no_source.copy()

        # Define nodal power vector candidate to the desired nodal power vector.
        node_power_vector_wye_candidate_no_source = node_power_vector_wye_no_source.copy()
        node_power_vector_delta_candidate_no_source = node_power_vector_delta_no_source.copy()

        # Instantiate outer iteration variables.
        is_final = False
        outer_iteration = 0

        # Outer iteration between power vector candidate selection and fixed point voltage solution algorithm
        # until a final solution is found.
        while ~is_final & (outer_iteration < outer_iteration_limit):

            # Outer solution algorithm based on fixed-point solution conditions check.
            # - Checks solution conditions and adjust power vector candidate if necessary, before solving for voltage.
            if outer_solution_algorithm == "check_conditions":

                # Reset nodal power vector candidate to the desired nodal power vector.
                node_power_vector_wye_candidate_no_source = node_power_vector_wye_no_source.copy()
                node_power_vector_delta_candidate_no_source = node_power_vector_delta_no_source.copy()

                # Check solution conditions for nodal power vector candidate.
                is_final = PowerFlowSolutionFixedPoint.check_solution_conditions(
                    electric_grid_model,
                    node_power_vector_wye_initial_no_source,
                    node_power_vector_delta_initial_no_source,
                    node_power_vector_wye_candidate_no_source,
                    node_power_vector_delta_candidate_no_source,
                    node_voltage_vector_initial_no_source,
                )

                # Instantiate power candidate iteration variable.
                power_candidate_iteration = 0
                is_valid = is_final.copy()

                # If solution conditions are violated, iteratively reduce power to find a power vector candidate
                # which satisfies the solution conditions.
                while ~is_valid & (power_candidate_iteration < power_candidate_iteration_limit):

                    # Reduce nodal power vector candidate.
                    node_power_vector_wye_candidate_no_source -= power_candidate_reduction_factor * (
                        node_power_vector_wye_candidate_no_source - node_power_vector_wye_initial_no_source
                    )
                    node_power_vector_delta_candidate_no_source -= power_candidate_reduction_factor * (
                        node_power_vector_delta_candidate_no_source - node_power_vector_delta_initial_no_source
                    )

                    is_valid = PowerFlowSolutionFixedPoint.check_solution_conditions(
                        electric_grid_model,
                        node_power_vector_wye_initial_no_source,
                        node_power_vector_delta_initial_no_source,
                        node_power_vector_wye_candidate_no_source,
                        node_power_vector_delta_candidate_no_source,
                        node_voltage_vector_initial_no_source,
                    )
                    power_candidate_iteration += 1

                # Reaching the iteration limit is considered undesired and triggers a warning.
                if power_candidate_iteration >= power_candidate_iteration_limit:
                    logger.warning(
                        "Power vector candidate selection algorithm for fixed-point solution reached "
                        f"maximum limit of {power_candidate_iteration_limit} iterations."
                    )

                # Store current candidate power vectors as initial power vectors
                # for next round of computation of solution conditions.
                node_power_vector_wye_initial_no_source = node_power_vector_wye_candidate_no_source.copy()
                node_power_vector_delta_initial_no_source = node_power_vector_delta_candidate_no_source.copy()

            # Instantiate fixed point iteration variables.
            voltage_iteration = 0
            voltage_change = np.inf
            while (voltage_iteration < voltage_iteration_limit) & (voltage_change > voltage_tolerance):

                # Calculate fixed point equation.
                node_voltage_vector_estimate_no_source = (
                    np.transpose([electric_grid_model.node_voltage_vector_reference_no_source])
                    + np.transpose(
                        [
                            scipy.sparse.linalg.spsolve(
                                electric_grid_model.node_admittance_matrix_no_source,
                                (
                                    (
                                        (np.conj(np.transpose([node_voltage_vector_initial_no_source])) ** -1)
                                        * np.conj(np.transpose([node_power_vector_wye_candidate_no_source]))
                                    )
                                    + (
                                        np.transpose(electric_grid_model.node_transformation_matrix_no_source)
                                        @ (
                                            (
                                                (
                                                    electric_grid_model.node_transformation_matrix_no_source
                                                    @ np.conj(np.transpose([node_voltage_vector_initial_no_source]))
                                                )
                                                ** -1
                                            )
                                            * np.conj(np.transpose([node_power_vector_delta_candidate_no_source]))
                                        )
                                    )
                                ),
                            )
                        ]
                    )
                ).ravel()

                # Calculate voltage change from previous iteration.
                voltage_change = np.max(
                    np.abs(node_voltage_vector_estimate_no_source - node_voltage_vector_initial_no_source)
                )

                # Set voltage solution as initial voltage for next iteration.
                node_voltage_vector_initial_no_source = node_voltage_vector_estimate_no_source.copy()

                # Increment voltage iteration counter.
                voltage_iteration += 1

            # Outer solution algorithm based on voltage solution check.
            # - Checks if voltage solution exceeded iteration limit and adjusts power vector candidate if needed.
            if outer_solution_algorithm == "check_solution":

                # If voltage solution exceeds iteration limit, reduce power and re-try voltage solution.
                if voltage_iteration >= voltage_iteration_limit:

                    # Reduce nodal power vector candidate.
                    node_power_vector_wye_candidate_no_source *= power_candidate_reduction_factor
                    node_power_vector_delta_candidate_no_source *= power_candidate_reduction_factor

                    # Reset initial nodal voltage vector.
                    node_voltage_vector_initial_no_source = (
                        electric_grid_model.node_voltage_vector_reference_no_source.copy()
                    )

                # Otherwise, if power has previously been reduced, raise back power and re-try voltage solution.
                else:
                    if (node_power_vector_wye_candidate_no_source != node_power_vector_wye_no_source).any() or (
                        node_power_vector_delta_candidate_no_source != node_power_vector_delta_no_source
                    ).any():

                        # Increase nodal power vector candidate.
                        node_power_vector_wye_candidate_no_source *= power_candidate_reduction_factor**-1
                        node_power_vector_delta_candidate_no_source *= power_candidate_reduction_factor**-1

                    else:
                        is_final = True

            # For fixed-point algorithm, reaching the iteration limit is considered undesired and triggers a warning
            elif voltage_iteration >= voltage_iteration_limit:
                logger.warning(
                    "Fixed point voltage solution algorithm reached "
                    f"maximum limit of {voltage_iteration_limit} iterations."
                )

            # Increment outer iteration counter.
            outer_iteration += 1

        # Reaching the outer iteration limit is considered undesired and triggers a warning.
        if outer_iteration >= outer_iteration_limit:
            logger.warning(
                "Outer wrapper algorithm for fixed-point solution reached "
                f"maximum limit of {outer_iteration_limit} iterations."
            )

        # Debug message.
        logger.debug("Completed fixed point solution algorithm. " f"Outer wrapper iterations: {outer_iteration}")

        # Get full voltage vector.
        node_voltage_vector = np.zeros(len(electric_grid_model.nodes), dtype=complex)
        node_voltage_vector[
            mesmo.utils.get_index(electric_grid_model.nodes, node_type="source")
        ] += electric_grid_model.node_voltage_vector_reference_source
        node_voltage_vector[
            mesmo.utils.get_index(electric_grid_model.nodes, node_type="no_source")
        ] += node_voltage_vector_initial_no_source  # Takes value of `node_voltage_vector_estimate_no_source`.

        return node_voltage_vector

    @staticmethod
    def get_branch_power(electric_grid_model: ElectricGridModel, node_voltage_vector: np.ndarray):
        """Get branch power vectors by calculating power flow with given nodal voltage.

        - Returns two branch power vectors, where `branch_power_vector_1` represents the
          "from"-direction and `branch_power_vector_2` represents the "to"-direction.
        """

        # Obtain branch admittance and incidence matrices.
        branch_admittance_1_matrix = electric_grid_model.branch_admittance_1_matrix
        branch_admittance_2_matrix = electric_grid_model.branch_admittance_2_matrix
        branch_incidence_1_matrix = electric_grid_model.branch_incidence_1_matrix
        branch_incidence_2_matrix = electric_grid_model.branch_incidence_2_matrix

        # Calculate branch power vectors.
        branch_power_vector_1 = (
            (branch_incidence_1_matrix @ np.transpose([node_voltage_vector]))
            * np.conj(branch_admittance_1_matrix @ np.transpose([node_voltage_vector]))
        ).ravel()
        branch_power_vector_2 = (
            (branch_incidence_2_matrix @ np.transpose([node_voltage_vector]))
            * np.conj(branch_admittance_2_matrix @ np.transpose([node_voltage_vector]))
        ).ravel()

        # Make modifications for single-phase-equivalent modelling.
        if electric_grid_model.is_single_phase_equivalent:
            branch_power_vector_1 *= 3
            branch_power_vector_2 *= 3

        return (branch_power_vector_1, branch_power_vector_2)

    @staticmethod
    def get_loss(electric_grid_model: ElectricGridModel, node_voltage_vector: np.ndarray):
        """Get total electric losses with given nodal voltage."""

        # Calculate total losses.
        # TODO: Check if summing up branch power is faster.
        # loss = (
        #     np.sum(
        #         branch_power_vector_1
        #         + branch_power_vector_2
        #     )
        # )
        loss = (
            np.array([node_voltage_vector])
            @ np.conj(electric_grid_model.node_admittance_matrix)
            @ np.transpose([np.conj(node_voltage_vector)])
        ).ravel()

        # Make modifications for single-phase-equivalent modelling.
        if electric_grid_model.is_single_phase_equivalent:
            loss *= 3

        return loss


class PowerFlowSolutionZBus(PowerFlowSolutionFixedPoint):
    """Implicit Z-bus power flow solution object."""

    # Overwrite `check_solution_conditions`, which is invalid for the Z-bus power flow.
    @staticmethod
    def check_solution_conditions(*args, **kwargs):
        raise NotImplementedError("This method is invalid for the Z-bus power flow.")

    @staticmethod
    def get_voltage(
        electric_grid_model: ElectricGridModel,
        der_power_vector: np.ndarray,
        voltage_iteration_limit=100,
        voltage_tolerance=1e-2,
        **kwargs,
    ) -> np.ndarray:
        """Get nodal voltage vector by solving with the implicit Z-bus method."""

        # Implicit Z-bus power flow solution (Arif Ahmed).
        # - “Can, Can, Lah!” (literal meaning, can accomplish)
        # - <https://www.financialexpress.com/opinion/singapore-turns-50-the-remarkable-nation-that-can-lah/115775/>

        # Obtain nodal power vectors.
        node_power_vector_wye_no_source = (
            electric_grid_model.der_incidence_wye_matrix_no_source @ np.transpose([der_power_vector])
        ).ravel()
        node_power_vector_delta_no_source = (
            electric_grid_model.der_incidence_delta_matrix_no_source @ np.transpose([der_power_vector])
        ).ravel()

        # Obtain utility variables.
        node_admittance_matrix_no_source_inverse = scipy.sparse.linalg.inv(
            electric_grid_model.node_admittance_matrix_no_source.tocsc()
        )
        node_admittance_matrix_source_to_no_source = electric_grid_model.node_admittance_matrix[
            np.ix_(
                mesmo.utils.get_index(electric_grid_model.nodes, node_type="no_source"),
                mesmo.utils.get_index(electric_grid_model.nodes, node_type="source"),
            )
        ]
        node_voltage_vector_initial_no_source = electric_grid_model.node_voltage_vector_reference_no_source.copy()

        # Instantiate implicit Z-bus power flow iteration variables.
        voltage_iteration = 0
        voltage_change = np.inf
        while (voltage_iteration < voltage_iteration_limit) & (voltage_change > voltage_tolerance):

            # Calculate current injections.
            node_current_injection_delta_in_wye_no_source = (
                electric_grid_model.node_transformation_matrix_no_source.transpose()
                @ np.conj(
                    np.linalg.inv(
                        np.diag(
                            (
                                electric_grid_model.node_transformation_matrix_no_source
                                @ node_voltage_vector_initial_no_source
                            ).ravel()
                        )
                    )
                    @ node_power_vector_wye_no_source
                )
            )
            node_current_injection_wye_no_source = np.conj(node_power_vector_delta_no_source) / np.conj(
                node_voltage_vector_initial_no_source
            )
            node_current_injection_no_source = (
                node_current_injection_delta_in_wye_no_source + node_current_injection_wye_no_source
            )

            # Calculate voltage.
            node_voltage_vector_estimate_no_source = node_admittance_matrix_no_source_inverse @ (
                -node_admittance_matrix_source_to_no_source @ electric_grid_model.node_voltage_vector_reference_source
                + node_current_injection_no_source
            )
            # node_voltage_vector_estimate_no_source = (
            #     electric_grid_model.node_voltage_vector_reference_no_source
            #     + node_admittance_matrix_no_source_inverse @ node_current_injection_no_source
            # )

            # Calculate voltage change from previous iteration.
            voltage_change = np.max(
                np.abs(node_voltage_vector_estimate_no_source - node_voltage_vector_initial_no_source)
            )

            # Set voltage estimate as new initial voltage for next iteration.
            node_voltage_vector_initial_no_source = node_voltage_vector_estimate_no_source.copy()

            # Increment voltage iteration counter.
            voltage_iteration += 1

        # Reaching the iteration limit is considered undesired and triggers a warning.
        if voltage_iteration >= voltage_iteration_limit:
            logger.warning(
                "Z-bus solution algorithm reached " f"maximum limit of {voltage_iteration_limit} iterations."
            )

        # Get full voltage vector.
        node_voltage_vector = np.zeros(len(electric_grid_model.nodes), dtype=complex)
        node_voltage_vector[
            mesmo.utils.get_index(electric_grid_model.nodes, node_type="source")
        ] += electric_grid_model.node_voltage_vector_reference_source
        node_voltage_vector[
            mesmo.utils.get_index(electric_grid_model.nodes, node_type="no_source")
        ] += node_voltage_vector_initial_no_source  # Takes value of `node_voltage_vector_estimate_no_source`.

        return node_voltage_vector


class PowerFlowSolutionOpenDSS(PowerFlowSolutionBase):
    """OpenDSS power flow solution object."""

    @multimethod.multimethod
    def __init__(self, scenario_name: str, **kwargs):

        # Obtain `electric_grid_model`.
        electric_grid_model = ElectricGridModelOpenDSS(scenario_name)

        self.__init__(electric_grid_model, **kwargs)

    @multimethod.multimethod
    def __init__(self, electric_grid_model: ElectricGridModelOpenDSS, **kwargs):

        # Obtain `der_power_vector`, assuming nominal power conditions.
        der_power_vector = electric_grid_model.der_power_vector_reference

        self.__init__(electric_grid_model, der_power_vector, **kwargs)

    @multimethod.multimethod
    def __init__(self, electric_grid_model: ElectricGridModelOpenDSS, der_power_vector: np.ndarray, **kwargs):

        # Store DER power vector.
        self.der_power_vector = der_power_vector.ravel()

        # Check if correct OpenDSS circuit is initialized, otherwise reinitialize.
        if opendssdirect.Circuit.Name() != electric_grid_model.circuit_name:
            electric_grid_model.__init__(electric_grid_model.electric_grid_data)

        # Set DER power vector in OpenDSS model.
        for der_index, der_name in enumerate(electric_grid_model.der_names):
            # TODO: For OpenDSS, all DERs are assumed to be loads.
            opendss_command_string = (
                f"load.{der_name}.kw = {- np.real(self.der_power_vector[der_index]) / 1000.0}"
                + f"\nload.{der_name}.kvar = {- np.imag(self.der_power_vector[der_index]) / 1000.0}"
            )
            logger.debug(f"opendss_command_string = \n{opendss_command_string}")
            opendssdirect.run_command(opendss_command_string)

        # Solve OpenDSS model.
        opendssdirect.run_command("solve")

        # Obtain voltage solution.
        self.node_voltage_vector = self.get_voltage(electric_grid_model)

        # Obtain branch flow solution.
        (self.branch_power_vector_1, self.branch_power_vector_2) = self.get_branch_power()

        # Obtain loss solution.
        self.loss = self.get_loss()

    @staticmethod
    def get_voltage(electric_grid_model: ElectricGridModelOpenDSS):
        """Get nodal voltage vector by solving OpenDSS model.

        - OpenDSS model must be readily set up, with the desired power being set for all DERs.
        """

        # Create index for OpenDSS nodes.
        opendss_nodes = pd.Series(opendssdirect.Circuit.AllNodeNames()).str.split(".", expand=True)
        opendss_nodes.columns = ["node_name", "phase"]
        opendss_nodes.loc[:, "phase"] = opendss_nodes.loc[:, "phase"].astype(int)
        opendss_nodes = pd.MultiIndex.from_frame(opendss_nodes)

        # Extract nodal voltage vector and reindex to match MESMO nodes order.
        node_voltage_vector_solution = (
            pd.Series(
                (
                    np.array(opendssdirect.Circuit.AllBusVolts()[0::2])
                    + 1j * np.array(opendssdirect.Circuit.AllBusVolts()[1::2])
                ),
                index=opendss_nodes,
            )
            .reindex(electric_grid_model.nodes.droplevel("node_type"))
            .values
        )

        # Make modifications for single-phase-equivalent modelling.
        if electric_grid_model.is_single_phase_equivalent:
            node_voltage_vector_solution /= np.sqrt(3)

        return node_voltage_vector_solution

    @staticmethod
    def get_branch_power():
        """Get branch power vectors by solving OpenDSS model.

        - OpenDSS model must be readily set up, with the desired power being set for all DERs.
        """

        # Solve OpenDSS model.
        opendssdirect.run_command("solve")

        # Instantiate branch vectors.
        branch_power_vector_1 = np.full(
            ((opendssdirect.Lines.Count() + opendssdirect.Transformers.Count()), 3), np.nan, dtype=complex
        )
        branch_power_vector_2 = np.full(
            ((opendssdirect.Lines.Count() + opendssdirect.Transformers.Count()), 3), np.nan, dtype=complex
        )

        # Instantiate iteration variables.
        branch_vector_index = 0
        line_index = opendssdirect.Lines.First()

        # Obtain line branch power vectors.
        while line_index > 0:
            branch_power_opendss = np.array(opendssdirect.CktElement.Powers()) * 1000.0
            branch_phase_count = opendssdirect.CktElement.NumPhases()
            branch_power_vector_1[branch_vector_index, :branch_phase_count] = (
                branch_power_opendss[0 : (branch_phase_count * 2) : 2]
                + 1.0j * branch_power_opendss[1 : (branch_phase_count * 2) : 2]
            )
            branch_power_vector_2[branch_vector_index, :branch_phase_count] = (
                branch_power_opendss[0 + (branch_phase_count * 2) :: 2]
                + 1.0j * branch_power_opendss[1 + (branch_phase_count * 2) :: 2]
            )

            branch_vector_index += 1
            line_index = opendssdirect.Lines.Next()

        # Obtain transformer branch power vectors.
        transformer_index = opendssdirect.Transformers.First()
        while transformer_index > 0:
            branch_power_opendss = np.array(opendssdirect.CktElement.Powers()) * 1000.0
            branch_phase_count = opendssdirect.CktElement.NumPhases()
            skip_phase = 2 if 0 in opendssdirect.CktElement.NodeOrder() else 0  # Ignore ground nodes.
            branch_power_vector_1[branch_vector_index, :branch_phase_count] = (
                branch_power_opendss[0 : (branch_phase_count * 2) : 2]
                + 1.0j * branch_power_opendss[1 : (branch_phase_count * 2) : 2]
            )
            branch_power_vector_2[branch_vector_index, :branch_phase_count] = (
                branch_power_opendss[0 + (branch_phase_count * 2) + skip_phase : -skip_phase : 2]
                + 1.0j * branch_power_opendss[1 + (branch_phase_count * 2) + skip_phase : -skip_phase : 2]
            )

            branch_vector_index += 1
            transformer_index = opendssdirect.Transformers.Next()

        # Reshape branch power vectors to appropriate size and remove entries for nonexistent phases.
        # TODO: Sort vector by branch name if not in order.
        branch_power_vector_1 = branch_power_vector_1.flatten()
        branch_power_vector_2 = branch_power_vector_2.flatten()
        branch_power_vector_1 = branch_power_vector_1[~np.isnan(branch_power_vector_1)]
        branch_power_vector_2 = branch_power_vector_2[~np.isnan(branch_power_vector_2)]

        return (branch_power_vector_1, branch_power_vector_2)

    @staticmethod
    def get_loss():
        """Get total loss by solving OpenDSS model.

        - OpenDSS model must be readily set up, with the desired power being set for all DERs.
        """

        # Solve OpenDSS model.
        opendssdirect.run_command("solve")

        # Obtain loss.
        loss = opendssdirect.Circuit.Losses()[0] + 1.0j * opendssdirect.Circuit.Losses()[1]

        return loss


class PowerFlowSolution(PowerFlowSolutionFixedPoint):
    """Electric power flow solution object. This object is a wrapper for the default power flow solution method
    as defined by inheritance. Currently, `PowerFlowSolutionFixedPoint` is the default method for solving the
    electric grid power flow.
    """


class PowerFlowSolutionSet(mesmo.utils.ObjectBase):

    power_flow_solutions: typing.Dict[pd.Timestamp, PowerFlowSolutionBase]
    electric_grid_model: ElectricGridModel
    der_power_vector: pd.DataFrame
    timesteps: pd.Index

    @multimethod.multimethod
    def __init__(
        self,
        electric_grid_model: ElectricGridModel,
        der_operation_results: ElectricGridDEROperationResults,
        **kwargs,
    ):

        der_power_vector = (
            der_operation_results.der_active_power_vector + 1.0j * der_operation_results.der_reactive_power_vector
        )

        self.__init__(electric_grid_model, der_power_vector, **kwargs)

    @multimethod.multimethod
    def __init__(
        self,
        electric_grid_model: ElectricGridModel,
        der_power_vector: pd.DataFrame,
        power_flow_solution_method=PowerFlowSolutionFixedPoint,
    ):

        # Store attributes.
        self.electric_grid_model = electric_grid_model
        self.der_power_vector = der_power_vector
        self.timesteps = self.electric_grid_model.timesteps

        # Obtain power flow solutions.
        power_flow_solutions = mesmo.utils.starmap(
            power_flow_solution_method, zip(itertools.repeat(self.electric_grid_model), der_power_vector.values)
        )
        self.power_flow_solutions = dict(zip(self.timesteps, power_flow_solutions))

    def get_results(self) -> ElectricGridOperationResults:

        # Instantiate results variables.
        der_power_vector = pd.DataFrame(columns=self.electric_grid_model.ders, index=self.timesteps, dtype=complex)
        node_voltage_vector = pd.DataFrame(columns=self.electric_grid_model.nodes, index=self.timesteps, dtype=complex)
        branch_power_vector_1 = pd.DataFrame(
            columns=self.electric_grid_model.branches, index=self.timesteps, dtype=complex
        )
        branch_power_vector_2 = pd.DataFrame(
            columns=self.electric_grid_model.branches, index=self.timesteps, dtype=complex
        )
        loss = pd.DataFrame(columns=["total"], index=self.timesteps, dtype=complex)

        # Obtain results.
        for timestep in self.timesteps:
            power_flow_solution = self.power_flow_solutions[timestep]
            der_power_vector.loc[timestep, :] = power_flow_solution.der_power_vector
            node_voltage_vector.loc[timestep, :] = power_flow_solution.node_voltage_vector
            branch_power_vector_1.loc[timestep, :] = power_flow_solution.branch_power_vector_1
            branch_power_vector_2.loc[timestep, :] = power_flow_solution.branch_power_vector_2
            loss.loc[timestep, :] = power_flow_solution.loss
        der_active_power_vector = der_power_vector.apply(np.real)
        der_reactive_power_vector = der_power_vector.apply(np.imag)
        node_voltage_magnitude_vector = np.abs(node_voltage_vector)
        branch_power_magnitude_vector_1 = np.abs(branch_power_vector_1)
        branch_power_magnitude_vector_2 = np.abs(branch_power_vector_2)
        loss_active = loss.apply(np.real)
        loss_reactive = loss.apply(np.imag)

        # Obtain per-unit values.
        der_active_power_vector_per_unit = der_active_power_vector * mesmo.utils.get_inverse_with_zeros(
            np.real(self.electric_grid_model.der_power_vector_reference)
        )
        der_reactive_power_vector_per_unit = der_reactive_power_vector * mesmo.utils.get_inverse_with_zeros(
            np.imag(self.electric_grid_model.der_power_vector_reference)
        )
        node_voltage_magnitude_vector_per_unit = node_voltage_magnitude_vector * mesmo.utils.get_inverse_with_zeros(
            np.abs(self.electric_grid_model.node_voltage_vector_reference)
        )
        branch_power_magnitude_vector_1_per_unit = branch_power_magnitude_vector_1 * mesmo.utils.get_inverse_with_zeros(
            self.electric_grid_model.branch_power_vector_magnitude_reference
        )
        branch_power_magnitude_vector_2_per_unit = branch_power_magnitude_vector_2 * mesmo.utils.get_inverse_with_zeros(
            self.electric_grid_model.branch_power_vector_magnitude_reference
        )

        # Store results.
        return ElectricGridOperationResults(
            electric_grid_model=self.electric_grid_model,
            der_active_power_vector=der_active_power_vector,
            der_active_power_vector_per_unit=der_active_power_vector_per_unit,
            der_reactive_power_vector=der_reactive_power_vector,
            der_reactive_power_vector_per_unit=der_reactive_power_vector_per_unit,
            node_voltage_magnitude_vector=node_voltage_magnitude_vector,
            node_voltage_magnitude_vector_per_unit=node_voltage_magnitude_vector_per_unit,
            branch_power_magnitude_vector_1=branch_power_magnitude_vector_1,
            branch_power_magnitude_vector_1_per_unit=branch_power_magnitude_vector_1_per_unit,
            branch_power_magnitude_vector_2=branch_power_magnitude_vector_2,
            branch_power_magnitude_vector_2_per_unit=branch_power_magnitude_vector_2_per_unit,
            loss_active=loss_active,
            loss_reactive=loss_reactive,
        )


class LinearElectricGridModelBase(mesmo.utils.ObjectBase):
    """Abstract linear electric model object, consisting of the sensitivity matrices for
    voltage / voltage magnitude / squared branch power / active loss / reactive loss by changes in nodal wye power /
    nodal delta power.

    Note:
        This abstract class only defines the expected variables of linear electric grid model objects,
        but does not implement any functionality.

    Attributes:
        electric_grid_model (ElectricGridModel): Electric grid model object.
        power_flow_solution (PowerFlowSolutionBase): Reference power flow solution object.
        sensitivity_voltage_by_power_wye_active (sp.spmatrix): Sensitivity matrix for complex voltage vector
            by active wye power vector.
        sensitivity_voltage_by_power_wye_reactive (sp.spmatrix): Sensitivity matrix for complex voltage
            vector by reactive wye power vector.
        sensitivity_voltage_by_power_delta_active (sp.spmatrix): Sensitivity matrix for complex voltage vector
            by active delta power vector.
        sensitivity_voltage_by_power_delta_reactive (sp.spmatrix): Sensitivity matrix for complex voltage
            vector by reactive delta power vector.
        sensitivity_voltage_by_der_power_active (sp.spmatrix): Sensitivity matrix for
            complex voltage vector by DER active power vector.
        sensitivity_voltage_by_der_power_reactive (sp.spmatrix): Sensitivity matrix for
            complex voltage vector by DER reactive power vector.
        sensitivity_voltage_magnitude_by_power_wye_active (sp.spmatrix): Sensitivity matrix for voltage
            magnitude vector by active wye power vector.
        sensitivity_voltage_magnitude_by_power_wye_reactive (sp.spmatrix): Sensitivity matrix for
            voltage magnitude vector by reactive wye power vector.
        sensitivity_voltage_magnitude_by_power_delta_active (sp.spmatrix): Sensitivity matrix for
            voltage magnitude vector by active delta power vector.
        sensitivity_voltage_magnitude_by_power_delta_reactive (sp.spmatrix): Sensitivity matrix for
            voltage magnitude vector by reactive delta power vector.
        sensitivity_voltage_magnitude_by_der_power_active (sp.spmatrix): Sensitivity matrix for
            voltage magnitude vector by DER active power vector.
        sensitivity_voltage_magnitude_by_der_power_reactive (sp.spmatrix): Sensitivity matrix for
            voltage magnitude vector by DER reactive power vector.
        sensitivity_branch_power_1_by_power_wye_active (sp.spmatrix): Sensitivity matrix for
            complex branch flow power vector 1 ('from' direction) by active wye power vector.
        sensitivity_branch_power_1_by_power_wye_reactive (sp.spmatrix): Sensitivity matrix for
            complex branch flow power vector 1 ('from' direction) by reactive wye power vector.
        sensitivity_branch_power_1_by_power_delta_active (sp.spmatrix): Sensitivity matrix for
            complex branch flow power vector 1 ('from' direction) by active delta power vector.
        sensitivity_branch_power_1_by_power_delta_reactive (sp.spmatrix): Sensitivity matrix for
            complex branch flow power vector 1 ('from' direction) by reactive delta power vector.
        sensitivity_branch_power_1_by_der_power_active (sp.spmatrix): Sensitivity matrix for
            complex branch flow power vector 1 by DER active power vector.
        sensitivity_branch_power_1_by_der_power_reactive (sp.spmatrix): Sensitivity matrix for
            complex branch flow power vector 1 by DER reactive power vector.
        sensitivity_branch_power_2_by_power_wye_active (sp.spmatrix): Sensitivity matrix for
            complex branch flow power vector 2 ('to' direction) by active wye power vector.
        sensitivity_branch_power_2_by_power_wye_reactive (sp.spmatrix): Sensitivity matrix for
            complex branch flow power vector 2 ('to' direction) by reactive wye power vector.
        sensitivity_branch_power_2_by_power_delta_active (sp.spmatrix): Sensitivity matrix for
            complex branch flow power vector 2 ('to' direction) by active delta power vector.
        sensitivity_branch_power_2_by_power_delta_reactive (sp.spmatrix): Sensitivity matrix for
            complex branch flow power vector 2 ('to' direction) by reactive delta power vector.
        sensitivity_branch_power_2_by_der_power_active (sp.spmatrix): Sensitivity matrix for
            complex branch flow power vector 2 by DER active power vector.
        sensitivity_branch_power_2_by_der_power_reactive (sp.spmatrix): Sensitivity matrix for
            complex branch flow power vector 2 by DER reactive power vector.
        sensitivity_branch_power_1_magnitude_by_power_wye_active (sp.spmatrix): Sensitivity matrix for
            branch flow power magnitude vector 1 ('from' direction) by active wye power vector.
        sensitivity_branch_power_1_magnitude_by_power_wye_reactive (sp.spmatrix): Sensitivity matrix for
            branch flow power magnitude vector 1 ('from' direction) by reactive wye power vector.
        sensitivity_branch_power_1_magnitude_by_power_delta_active (sp.spmatrix): Sensitivity matrix for
            branch flow power magnitude vector 1 ('from' direction) by active delta power vector.
        sensitivity_branch_power_1_magnitude_by_power_delta_reactive (sp.spmatrix): Sensitivity matrix for
            branch flow power magnitude vector 1 ('from' direction) by reactive delta power vector.
        sensitivity_branch_power_1_magnitude_by_der_power_active (sp.spmatrix): Sensitivity matrix for
            branch flow power magnitude vector 1 by DER active power vector.
        sensitivity_branch_power_1_magnitude_by_der_power_reactive (sp.spmatrix): Sensitivity matrix for
            branch flow power magnitude vector 1 by DER reactive power vector.
        sensitivity_branch_power_2_magnitude_by_power_wye_active (sp.spmatrix): Sensitivity matrix for
            branch flow power magnitude vector 2 ('to' direction) by active wye power vector.
        sensitivity_branch_power_2_magnitude_by_power_wye_reactive (sp.spmatrix): Sensitivity matrix for
            branch flow power magnitude vector 2 ('to' direction) by reactive wye power vector.
        sensitivity_branch_power_2_magnitude_by_power_delta_active (sp.spmatrix): Sensitivity matrix for
            branch flow power magnitude vector 2 ('to' direction) by active delta power vector.
        sensitivity_branch_power_2_magnitude_by_power_delta_reactive (sp.spmatrix): Sensitivity matrix for
            branch flow power magnitude vector 2 ('to' direction) by reactive delta power vector.
        sensitivity_branch_power_2_magnitude_by_der_power_active (sp.spmatrix): Sensitivity matrix for
            branch flow power magnitude vector 2 by DER active power vector.
        sensitivity_branch_power_2_magnitude_by_der_power_reactive (sp.spmatrix): Sensitivity matrix for
            branch flow power magnitude vector 2 by DER reactive power vector.
        sensitivity_loss_active_by_power_wye_active (sp.spmatrix): Sensitivity matrix for
            active loss by active wye power vector.
        sensitivity_loss_active_by_power_wye_reactive (sp.spmatrix): Sensitivity matrix for
            active loss by reactive wye power vector.
        sensitivity_loss_active_by_power_delta_active (sp.spmatrix): Sensitivity matrix for
            active loss by active delta power vector.
        sensitivity_loss_active_by_power_delta_reactive (sp.spmatrix): Sensitivity matrix for
            active loss by reactive delta power vector.
        sensitivity_loss_active_by_der_power_active (sp.spmatrix): Sensitivity matrix for
            active loss by DER active power vector.
        sensitivity_loss_active_by_der_power_reactive (sp.spmatrix): Sensitivity matrix for
            active loss by DER reactive power vector.
        sensitivity_loss_reactive_by_power_wye_active (sp.spmatrix): Sensitivity matrix for
            reactive loss by active wye power vector.
        sensitivity_loss_reactive_by_power_wye_reactive (sp.spmatrix): Sensitivity matrix for
            reactive loss by reactive wye power vector.
        sensitivity_loss_reactive_by_power_delta_active (sp.spmatrix): Sensitivity matrix for
            reactive loss by active delta power vector.
        sensitivity_loss_reactive_by_power_delta_reactive (sp.spmatrix): Sensitivity matrix for
            reactive loss by reactive delta power vector.
        sensitivity_loss_reactive_by_der_power_active (sp.spmatrix): Sensitivity matrix for
            reactive loss by DER active power vector.
        sensitivity_loss_reactive_by_der_power_reactive (sp.spmatrix): Sensitivity matrix for
            reactive loss by DER reactive power vector.
    """

    electric_grid_model: ElectricGridModel
    power_flow_solution: PowerFlowSolutionBase
    sensitivity_voltage_by_power_wye_active: sp.spmatrix
    sensitivity_voltage_by_power_wye_reactive: sp.spmatrix
    sensitivity_voltage_by_power_delta_active: sp.spmatrix
    sensitivity_voltage_by_power_delta_reactive: sp.spmatrix
    sensitivity_voltage_by_der_power_active: sp.spmatrix
    sensitivity_voltage_by_der_power_reactive: sp.spmatrix
    sensitivity_voltage_magnitude_by_power_wye_active: sp.spmatrix
    sensitivity_voltage_magnitude_by_power_wye_reactive: sp.spmatrix
    sensitivity_voltage_magnitude_by_power_delta_active: sp.spmatrix
    sensitivity_voltage_magnitude_by_power_delta_reactive: sp.spmatrix
    sensitivity_voltage_magnitude_by_der_power_active: sp.spmatrix
    sensitivity_voltage_magnitude_by_der_power_reactive: sp.spmatrix
    sensitivity_branch_power_1_by_power_wye_active: sp.spmatrix
    sensitivity_branch_power_1_by_power_wye_reactive: sp.spmatrix
    sensitivity_branch_power_1_by_power_delta_active: sp.spmatrix
    sensitivity_branch_power_1_by_power_delta_reactive: sp.spmatrix
    sensitivity_branch_power_1_by_der_power_active: sp.spmatrix
    sensitivity_branch_power_1_by_der_power_reactive: sp.spmatrix
    sensitivity_branch_power_2_by_power_wye_active: sp.spmatrix
    sensitivity_branch_power_2_by_power_wye_reactive: sp.spmatrix
    sensitivity_branch_power_2_by_power_delta_active: sp.spmatrix
    sensitivity_branch_power_2_by_power_delta_reactive: sp.spmatrix
    sensitivity_branch_power_2_by_der_power_active: sp.spmatrix
    sensitivity_branch_power_2_by_der_power_reactive: sp.spmatrix
    sensitivity_branch_power_1_magnitude_by_power_wye_active: sp.spmatrix
    sensitivity_branch_power_1_magnitude_by_power_wye_reactive: sp.spmatrix
    sensitivity_branch_power_1_magnitude_by_power_delta_active: sp.spmatrix
    sensitivity_branch_power_1_magnitude_by_power_delta_reactive: sp.spmatrix
    sensitivity_branch_power_1_magnitude_by_der_power_active: sp.spmatrix
    sensitivity_branch_power_1_magnitude_by_der_power_reactive: sp.spmatrix
    sensitivity_branch_power_2_magnitude_by_power_wye_active: sp.spmatrix
    sensitivity_branch_power_2_magnitude_by_power_wye_reactive: sp.spmatrix
    sensitivity_branch_power_2_magnitude_by_power_delta_active: sp.spmatrix
    sensitivity_branch_power_2_magnitude_by_power_delta_reactive: sp.spmatrix
    sensitivity_branch_power_2_magnitude_by_der_power_active: sp.spmatrix
    sensitivity_branch_power_2_magnitude_by_der_power_reactive: sp.spmatrix
    sensitivity_loss_active_by_power_wye_active: sp.spmatrix
    sensitivity_loss_active_by_power_wye_reactive: sp.spmatrix
    sensitivity_loss_active_by_power_delta_active: sp.spmatrix
    sensitivity_loss_active_by_power_delta_reactive: sp.spmatrix
    sensitivity_loss_active_by_der_power_active: sp.spmatrix
    sensitivity_loss_active_by_der_power_reactive: sp.spmatrix
    sensitivity_loss_reactive_by_power_wye_active: sp.spmatrix
    sensitivity_loss_reactive_by_power_wye_reactive: sp.spmatrix
    sensitivity_loss_reactive_by_power_delta_active: sp.spmatrix
    sensitivity_loss_reactive_by_power_delta_reactive: sp.spmatrix
    sensitivity_loss_reactive_by_der_power_active: sp.spmatrix
    sensitivity_loss_reactive_by_der_power_reactive: sp.spmatrix


class LinearElectricGridModelGlobal(LinearElectricGridModelBase):
    """Linear electric grid model object based on global approximations, consisting of the sensitivity matrices for
    voltage / voltage magnitude / squared branch power / active loss / reactive loss by changes in nodal wye power /
    nodal delta power.

    :syntax:
        - ``LinearElectricGridModelGlobal(electric_grid_model, power_flow_solution)``: Instantiate linear electric grid
          model object for given `electric_grid_model` and `power_flow_solution`.
        - ``LinearElectricGridModelGlobal(scenario_name)``: Instantiate linear electric grid model for given
          `scenario_name`. The required `electric_grid_model` is obtained for given `scenario_name` and the
          `power_flow_solution` is obtained for nominal power conditions.

    Parameters:
        electric_grid_model (ElectricGridModel): Electric grid model object.
        power_flow_solution (PowerFlowSolutionBase): Power flow solution object.
        scenario_name (str): MESMO scenario name.

    Attributes / variables are the same as in :class:`LinearElectricGridModelBase`.
    """

    @multimethod.multimethod
    def __init__(
        self,
        scenario_name: str,
    ):

        # Obtain electric grid model.
        electric_grid_model = ElectricGridModel(scenario_name)

        # Obtain der power vector.
        der_power_vector = electric_grid_model.der_power_vector_reference

        # Obtain power flow solution.
        power_flow_solution = PowerFlowSolutionFixedPoint(electric_grid_model, der_power_vector)

        self.__init__(electric_grid_model, power_flow_solution)

    @multimethod.multimethod
    def __init__(self, electric_grid_model: ElectricGridModel, power_flow_solution: PowerFlowSolutionBase):
        # TODO: Validate linear model with delta DERs.

        # Store power flow solution.
        self.power_flow_solution = power_flow_solution

        # Store electric grid model.
        self.electric_grid_model = electric_grid_model

        # Obtain shorthands for no-source matrices and vectors.
        electric_grid_model.node_admittance_matrix_no_source = electric_grid_model.node_admittance_matrix[
            np.ix_(
                mesmo.utils.get_index(electric_grid_model.nodes, node_type="no_source"),
                mesmo.utils.get_index(electric_grid_model.nodes, node_type="no_source"),
            )
        ]
        electric_grid_model.node_transformation_matrix_no_source = electric_grid_model.node_transformation_matrix[
            np.ix_(
                mesmo.utils.get_index(electric_grid_model.nodes, node_type="no_source"),
                mesmo.utils.get_index(electric_grid_model.nodes, node_type="no_source"),
            )
        ]
        node_voltage_no_source = self.power_flow_solution.node_voltage_vector[
            mesmo.utils.get_index(electric_grid_model.nodes, node_type="no_source")
        ]

        # Instantiate voltage sensitivity matrices.
        self.sensitivity_voltage_by_power_wye_active = sp.dok_matrix(
            (len(electric_grid_model.nodes), len(electric_grid_model.nodes)), dtype=complex
        )
        self.sensitivity_voltage_by_power_wye_reactive = sp.dok_matrix(
            (len(electric_grid_model.nodes), len(electric_grid_model.nodes)), dtype=complex
        )
        self.sensitivity_voltage_by_power_delta_active = sp.dok_matrix(
            (len(electric_grid_model.nodes), len(electric_grid_model.nodes)), dtype=complex
        )
        self.sensitivity_voltage_by_power_delta_reactive = sp.dok_matrix(
            (len(electric_grid_model.nodes), len(electric_grid_model.nodes)), dtype=complex
        )

        # Calculate voltage sensitivity matrices.
        # TODO: Document the change in sign in the reactive part.
        self.sensitivity_voltage_by_power_wye_active[
            np.ix_(
                mesmo.utils.get_index(electric_grid_model.nodes, node_type="no_source"),
                mesmo.utils.get_index(electric_grid_model.nodes, node_type="no_source"),
            )
        ] = scipy.sparse.linalg.spsolve(
            electric_grid_model.node_admittance_matrix_no_source.tocsc(),
            sp.diags(np.conj(node_voltage_no_source) ** -1, format="csc"),
        )
        self.sensitivity_voltage_by_power_wye_reactive[
            np.ix_(
                mesmo.utils.get_index(electric_grid_model.nodes, node_type="no_source"),
                mesmo.utils.get_index(electric_grid_model.nodes, node_type="no_source"),
            )
        ] = scipy.sparse.linalg.spsolve(
            1.0j * electric_grid_model.node_admittance_matrix_no_source.tocsc(),
            sp.diags(np.conj(node_voltage_no_source) ** -1, format="csc"),
        )
        self.sensitivity_voltage_by_power_delta_active[
            np.ix_(
                mesmo.utils.get_index(electric_grid_model.nodes, node_type="no_source"),
                mesmo.utils.get_index(electric_grid_model.nodes, node_type="no_source"),
            )
        ] = scipy.sparse.linalg.spsolve(
            electric_grid_model.node_admittance_matrix_no_source.tocsc(),
            np.transpose(electric_grid_model.node_transformation_matrix_no_source),
        ) @ sp.diags(
            ((electric_grid_model.node_transformation_matrix_no_source @ np.conj(node_voltage_no_source)) ** -1).ravel()
        )
        self.sensitivity_voltage_by_power_delta_reactive[
            np.ix_(
                mesmo.utils.get_index(electric_grid_model.nodes, node_type="no_source"),
                mesmo.utils.get_index(electric_grid_model.nodes, node_type="no_source"),
            )
        ] = scipy.sparse.linalg.spsolve(
            1.0j * electric_grid_model.node_admittance_matrix_no_source.tocsc(),
            np.transpose(electric_grid_model.node_transformation_matrix_no_source),
        ) @ sp.diags(
            ((electric_grid_model.node_transformation_matrix_no_source * np.conj(node_voltage_no_source)) ** -1).ravel()
        )

        self.sensitivity_voltage_by_der_power_active = (
            self.sensitivity_voltage_by_power_wye_active @ electric_grid_model.der_incidence_wye_matrix
            + self.sensitivity_voltage_by_power_delta_active @ electric_grid_model.der_incidence_delta_matrix
        )
        self.sensitivity_voltage_by_der_power_reactive = (
            self.sensitivity_voltage_by_power_wye_reactive @ electric_grid_model.der_incidence_wye_matrix
            + self.sensitivity_voltage_by_power_delta_reactive @ electric_grid_model.der_incidence_delta_matrix
        )

        self.sensitivity_voltage_magnitude_by_power_wye_active = sp.diags(
            abs(self.power_flow_solution.node_voltage_vector) ** -1
        ) @ np.real(
            sp.diags(np.conj(self.power_flow_solution.node_voltage_vector))
            @ self.sensitivity_voltage_by_power_wye_active
        )
        self.sensitivity_voltage_magnitude_by_power_wye_reactive = sp.diags(
            abs(self.power_flow_solution.node_voltage_vector) ** -1
        ) @ np.real(
            sp.diags(np.conj(self.power_flow_solution.node_voltage_vector))
            @ self.sensitivity_voltage_by_power_wye_reactive
        )
        self.sensitivity_voltage_magnitude_by_power_delta_active = sp.diags(
            abs(self.power_flow_solution.node_voltage_vector) ** -1
        ) @ np.real(
            sp.diags(np.conj(self.power_flow_solution.node_voltage_vector))
            @ self.sensitivity_voltage_by_power_delta_active
        )
        self.sensitivity_voltage_magnitude_by_power_delta_reactive = sp.diags(
            abs(self.power_flow_solution.node_voltage_vector) ** -1
        ) @ np.real(
            sp.diags(np.conj(self.power_flow_solution.node_voltage_vector))
            @ self.sensitivity_voltage_by_power_delta_reactive
        )

        self.sensitivity_voltage_magnitude_by_der_power_active = (
            self.sensitivity_voltage_magnitude_by_power_wye_active @ electric_grid_model.der_incidence_wye_matrix
            + self.sensitivity_voltage_magnitude_by_power_delta_active @ electric_grid_model.der_incidence_delta_matrix
        )
        self.sensitivity_voltage_magnitude_by_der_power_reactive = (
            self.sensitivity_voltage_magnitude_by_power_wye_reactive @ electric_grid_model.der_incidence_wye_matrix
            + self.sensitivity_voltage_magnitude_by_power_delta_reactive
            @ electric_grid_model.der_incidence_delta_matrix
        )

        # Calculate branch power sensitivity matrices.
        # TODO: Document the empirical fixes.
        sensitivity_branch_power_1_by_voltage_active = (
            sp.diags(
                np.conj(
                    self.electric_grid_model.branch_incidence_1_matrix
                    @ np.transpose([self.power_flow_solution.node_voltage_vector])
                ).ravel()
            )
            @ self.electric_grid_model.branch_admittance_1_matrix
        )
        sensitivity_branch_power_1_by_voltage_reactive = (
            sp.diags(
                -1.0
                * np.conj(
                    self.electric_grid_model.branch_incidence_1_matrix
                    @ np.transpose([self.power_flow_solution.node_voltage_vector])
                ).ravel()
            )
            @ self.electric_grid_model.branch_admittance_1_matrix
        )
        sensitivity_branch_power_2_by_voltage_active = (
            sp.diags(
                np.conj(
                    self.electric_grid_model.branch_incidence_2_matrix
                    @ np.transpose([self.power_flow_solution.node_voltage_vector])
                ).ravel()
            )
            @ self.electric_grid_model.branch_admittance_2_matrix
        )
        sensitivity_branch_power_2_by_voltage_reactive = (
            sp.diags(
                -1.0
                * np.conj(
                    self.electric_grid_model.branch_incidence_2_matrix
                    @ np.transpose([self.power_flow_solution.node_voltage_vector])
                ).ravel()
            )
            @ self.electric_grid_model.branch_admittance_2_matrix
        )

        self.sensitivity_branch_power_1_by_power_wye_active = np.conj(
            sensitivity_branch_power_1_by_voltage_active @ self.sensitivity_voltage_by_power_wye_active
        )
        self.sensitivity_branch_power_1_by_power_wye_reactive = -1.0 * np.conj(
            sensitivity_branch_power_1_by_voltage_reactive @ self.sensitivity_voltage_by_power_wye_reactive
        )
        self.sensitivity_branch_power_1_by_power_delta_active = np.conj(
            sensitivity_branch_power_1_by_voltage_active @ self.sensitivity_voltage_by_power_delta_active
        )
        self.sensitivity_branch_power_1_by_power_delta_reactive = -1.0 * np.conj(
            sensitivity_branch_power_1_by_voltage_reactive @ self.sensitivity_voltage_by_power_delta_reactive
        )
        self.sensitivity_branch_power_2_by_power_wye_active = np.conj(
            sensitivity_branch_power_2_by_voltage_active @ self.sensitivity_voltage_by_power_wye_active
        )
        self.sensitivity_branch_power_2_by_power_wye_reactive = -1.0 * np.conj(
            sensitivity_branch_power_2_by_voltage_reactive @ self.sensitivity_voltage_by_power_wye_reactive
        )
        self.sensitivity_branch_power_2_by_power_delta_active = np.conj(
            sensitivity_branch_power_2_by_voltage_active @ self.sensitivity_voltage_by_power_delta_active
        )
        self.sensitivity_branch_power_2_by_power_delta_reactive = -1.0 * np.conj(
            sensitivity_branch_power_2_by_voltage_reactive @ self.sensitivity_voltage_by_power_delta_reactive
        )

        self.sensitivity_branch_power_1_by_der_power_active = (
            self.sensitivity_branch_power_1_by_power_wye_active @ electric_grid_model.der_incidence_wye_matrix
            + self.sensitivity_branch_power_1_by_power_delta_active @ electric_grid_model.der_incidence_delta_matrix
        )
        self.sensitivity_branch_power_1_by_der_power_reactive = (
            self.sensitivity_branch_power_1_by_power_wye_reactive @ electric_grid_model.der_incidence_wye_matrix
            + self.sensitivity_branch_power_1_by_power_delta_reactive @ electric_grid_model.der_incidence_delta_matrix
        )
        self.sensitivity_branch_power_2_by_der_power_active = (
            self.sensitivity_branch_power_2_by_power_wye_active @ electric_grid_model.der_incidence_wye_matrix
            + self.sensitivity_branch_power_2_by_power_delta_active @ electric_grid_model.der_incidence_delta_matrix
        )
        self.sensitivity_branch_power_2_by_der_power_reactive = (
            self.sensitivity_branch_power_2_by_power_wye_reactive @ electric_grid_model.der_incidence_wye_matrix
            + self.sensitivity_branch_power_2_by_power_delta_reactive @ electric_grid_model.der_incidence_delta_matrix
        )

        self.sensitivity_branch_power_1_magnitude_by_power_wye_active = sp.diags(
            abs(self.power_flow_solution.branch_power_vector_1) ** -1
        ) @ np.real(
            sp.diags(np.conj(self.power_flow_solution.branch_power_vector_1))
            @ self.sensitivity_branch_power_1_by_power_wye_active
        )
        self.sensitivity_branch_power_1_magnitude_by_power_wye_reactive = sp.diags(
            abs(self.power_flow_solution.branch_power_vector_1) ** -1
        ) @ np.real(
            sp.diags(np.conj(self.power_flow_solution.branch_power_vector_1))
            @ self.sensitivity_branch_power_1_by_power_wye_reactive
        )
        self.sensitivity_branch_power_1_magnitude_by_power_delta_active = sp.diags(
            abs(self.power_flow_solution.branch_power_vector_1) ** -1
        ) @ np.real(
            sp.diags(np.conj(self.power_flow_solution.branch_power_vector_1))
            @ self.sensitivity_branch_power_1_by_power_delta_active
        )
        self.sensitivity_branch_power_1_magnitude_by_power_delta_reactive = sp.diags(
            abs(self.power_flow_solution.branch_power_vector_1) ** -1
        ) @ np.real(
            sp.diags(np.conj(self.power_flow_solution.branch_power_vector_1))
            @ self.sensitivity_branch_power_1_by_power_delta_reactive
        )
        self.sensitivity_branch_power_2_magnitude_by_power_wye_active = sp.diags(
            abs(self.power_flow_solution.branch_power_vector_2) ** -1
        ) @ np.real(
            sp.diags(np.conj(self.power_flow_solution.branch_power_vector_2))
            @ self.sensitivity_branch_power_2_by_power_wye_active
        )
        self.sensitivity_branch_power_2_magnitude_by_power_wye_reactive = sp.diags(
            abs(self.power_flow_solution.branch_power_vector_2) ** -1
        ) @ np.real(
            sp.diags(np.conj(self.power_flow_solution.branch_power_vector_2))
            @ self.sensitivity_branch_power_2_by_power_wye_reactive
        )
        self.sensitivity_branch_power_2_magnitude_by_power_delta_active = sp.diags(
            abs(self.power_flow_solution.branch_power_vector_2) ** -1
        ) @ np.real(
            sp.diags(np.conj(self.power_flow_solution.branch_power_vector_2))
            @ self.sensitivity_branch_power_2_by_power_delta_active
        )
        self.sensitivity_branch_power_2_magnitude_by_power_delta_reactive = sp.diags(
            abs(self.power_flow_solution.branch_power_vector_2) ** -1
        ) @ np.real(
            sp.diags(np.conj(self.power_flow_solution.branch_power_vector_2))
            @ self.sensitivity_branch_power_2_by_power_delta_reactive
        )

        self.sensitivity_branch_power_1_magnitude_by_der_power_active = (
            self.sensitivity_branch_power_1_magnitude_by_power_wye_active @ electric_grid_model.der_incidence_wye_matrix
            + self.sensitivity_branch_power_1_magnitude_by_power_delta_active
            @ electric_grid_model.der_incidence_delta_matrix
        )
        self.sensitivity_branch_power_1_magnitude_by_der_power_reactive = (
            self.sensitivity_branch_power_1_magnitude_by_power_wye_reactive
            @ electric_grid_model.der_incidence_wye_matrix
            + self.sensitivity_branch_power_1_magnitude_by_power_delta_reactive
            @ electric_grid_model.der_incidence_delta_matrix
        )
        self.sensitivity_branch_power_2_magnitude_by_der_power_active = (
            self.sensitivity_branch_power_2_magnitude_by_power_wye_active @ electric_grid_model.der_incidence_wye_matrix
            + self.sensitivity_branch_power_2_magnitude_by_power_delta_active
            @ electric_grid_model.der_incidence_delta_matrix
        )
        self.sensitivity_branch_power_2_magnitude_by_der_power_reactive = (
            self.sensitivity_branch_power_2_magnitude_by_power_wye_reactive
            @ electric_grid_model.der_incidence_wye_matrix
            + self.sensitivity_branch_power_2_magnitude_by_power_delta_reactive
            @ electric_grid_model.der_incidence_delta_matrix
        )

        # Calculate loss sensitivity matrices.
        self.sensitivity_loss_active_by_power_wye_active = sum(
            np.real(
                self.sensitivity_branch_power_1_by_power_wye_active
                + self.sensitivity_branch_power_2_by_power_wye_active
            )
        )
        self.sensitivity_loss_active_by_power_wye_reactive = sum(
            np.real(
                self.sensitivity_branch_power_1_by_power_wye_reactive
                + self.sensitivity_branch_power_2_by_power_wye_reactive
            )
        )
        self.sensitivity_loss_active_by_power_delta_active = sum(
            np.real(
                self.sensitivity_branch_power_1_by_power_delta_active
                + self.sensitivity_branch_power_2_by_power_delta_active
            )
        )
        self.sensitivity_loss_active_by_power_delta_reactive = sum(
            np.real(
                self.sensitivity_branch_power_1_by_power_delta_reactive
                + self.sensitivity_branch_power_2_by_power_delta_reactive
            )
        )
        self.sensitivity_loss_reactive_by_power_wye_active = sum(
            np.imag(
                self.sensitivity_branch_power_1_by_power_wye_active
                + self.sensitivity_branch_power_2_by_power_wye_active
            )
        )
        self.sensitivity_loss_reactive_by_power_wye_reactive = sum(
            np.imag(
                self.sensitivity_branch_power_1_by_power_wye_reactive
                + self.sensitivity_branch_power_2_by_power_wye_reactive
            )
        )
        self.sensitivity_loss_reactive_by_power_delta_active = sum(
            np.imag(
                self.sensitivity_branch_power_1_by_power_delta_active
                + self.sensitivity_branch_power_2_by_power_delta_active
            )
        )
        self.sensitivity_loss_reactive_by_power_delta_reactive = sum(
            np.imag(
                self.sensitivity_branch_power_1_by_power_delta_reactive
                + self.sensitivity_branch_power_2_by_power_delta_reactive
            )
        )

        self.sensitivity_loss_active_by_der_power_active = (
            self.sensitivity_loss_active_by_power_wye_active @ electric_grid_model.der_incidence_wye_matrix
            + self.sensitivity_loss_active_by_power_delta_active @ electric_grid_model.der_incidence_delta_matrix
        )
        self.sensitivity_loss_active_by_der_power_reactive = (
            self.sensitivity_loss_active_by_power_wye_reactive @ electric_grid_model.der_incidence_wye_matrix
            + self.sensitivity_loss_active_by_power_delta_reactive @ electric_grid_model.der_incidence_delta_matrix
        )
        self.sensitivity_loss_reactive_by_der_power_active = (
            self.sensitivity_loss_reactive_by_power_wye_active @ electric_grid_model.der_incidence_wye_matrix
            + self.sensitivity_loss_reactive_by_power_delta_active @ electric_grid_model.der_incidence_delta_matrix
        )
        self.sensitivity_loss_reactive_by_der_power_reactive = (
            self.sensitivity_loss_reactive_by_power_wye_reactive @ electric_grid_model.der_incidence_wye_matrix
            + self.sensitivity_loss_reactive_by_power_delta_reactive @ electric_grid_model.der_incidence_delta_matrix
        )


class LinearElectricGridModelLocal(LinearElectricGridModelBase):
    """Linear electric grid model object based on local approximations, consisting of the sensitivity matrices for
    voltage / voltage magnitude / squared branch power / active loss / reactive loss by changes in nodal wye power /
    nodal delta power.

    :syntax:
        - ``LinearElectricGridModelLocal(electric_grid_model, power_flow_solution)``: Instantiate linear electric grid
          model object for given `electric_grid_model` and `power_flow_solution`.
        - ``LinearElectricGridModelLocal(scenario_name)``: Instantiate linear electric grid model for given
          `scenario_name`. The required `electric_grid_model` is obtained for given `scenario_name` and the
          `power_flow_solution` is obtained for nominal power conditions.

    Parameters:
        electric_grid_model (ElectricGridModel): Electric grid model object.
        power_flow_solution (PowerFlowSolutionBase): Power flow solution object.
        scenario_name (str): MESMO scenario name.

    Attributes / variables are the same as in :class:`LinearElectricGridModelBase`.
    """

    @multimethod.multimethod
    def __init__(
        self,
        scenario_name: str,
    ):

        # Obtain electric grid model.
        electric_grid_model = ElectricGridModel(scenario_name)

        # Obtain der power vector.
        der_power_vector = electric_grid_model.der_power_vector_reference

        # Obtain power flow solution.
        power_flow_solution = PowerFlowSolutionFixedPoint(electric_grid_model, der_power_vector)

        self.__init__(electric_grid_model, power_flow_solution)

    @multimethod.multimethod
    def __init__(self, electric_grid_model: ElectricGridModel, power_flow_solution: PowerFlowSolutionBase):

        # Store power flow solution.
        self.power_flow_solution = power_flow_solution

        # Store electric grid model.
        self.electric_grid_model = electric_grid_model

        # Obtain shorthands for no-source matrices and vectors.
        electric_grid_model.node_admittance_matrix_no_source = electric_grid_model.node_admittance_matrix[
            np.ix_(
                mesmo.utils.get_index(electric_grid_model.nodes, node_type="no_source"),
                mesmo.utils.get_index(electric_grid_model.nodes, node_type="no_source"),
            )
        ]
        electric_grid_model.node_transformation_matrix_no_source = electric_grid_model.node_transformation_matrix[
            np.ix_(
                mesmo.utils.get_index(electric_grid_model.nodes, node_type="no_source"),
                mesmo.utils.get_index(electric_grid_model.nodes, node_type="no_source"),
            )
        ]
        node_voltage_no_source = self.power_flow_solution.node_voltage_vector[
            mesmo.utils.get_index(electric_grid_model.nodes, node_type="no_source")
        ]

        # Instantiate voltage sensitivity matrices.
        self.sensitivity_voltage_by_power_wye_active = sp.dok_matrix(
            (len(electric_grid_model.nodes), len(electric_grid_model.nodes)), dtype=complex
        )
        self.sensitivity_voltage_by_power_wye_reactive = sp.dok_matrix(
            (len(electric_grid_model.nodes), len(electric_grid_model.nodes)), dtype=complex
        )
        self.sensitivity_voltage_by_power_delta_active = sp.dok_matrix(
            (len(electric_grid_model.nodes), len(electric_grid_model.nodes)), dtype=complex
        )
        self.sensitivity_voltage_by_power_delta_reactive = sp.dok_matrix(
            (len(electric_grid_model.nodes), len(electric_grid_model.nodes)), dtype=complex
        )

        # Calculate utility matrices.
        A_matrix_inverse = sp.diags(
            (
                electric_grid_model.node_admittance_matrix_source_to_no_source
                @ electric_grid_model.node_voltage_vector_reference_source
                + electric_grid_model.node_admittance_matrix_no_source @ node_voltage_no_source
            )
            ** -1
        )
        A_matrix_conjugate = sp.diags(
            np.conj(
                electric_grid_model.node_admittance_matrix_source_to_no_source
                @ electric_grid_model.node_voltage_vector_reference_source
                + electric_grid_model.node_admittance_matrix_no_source @ node_voltage_no_source
            )
        )
        B_matrix = (
            A_matrix_conjugate
            - sp.diags(node_voltage_no_source)
            @ np.conj(electric_grid_model.node_admittance_matrix_no_source)
            @ A_matrix_inverse
            @ sp.diags(np.conj(node_voltage_no_source))
            @ electric_grid_model.node_admittance_matrix_no_source
        )

        # Calculate voltage sensitivity matrices.
        # - TODO: Consider delta loads.
        self.sensitivity_voltage_by_power_wye_active[
            np.ix_(
                mesmo.utils.get_index(electric_grid_model.nodes, node_type="no_source"),
                mesmo.utils.get_index(electric_grid_model.nodes, node_type="no_source"),
            )
        ] = scipy.sparse.linalg.spsolve(
            B_matrix.tocsc(),
            (
                sp.identity(len(node_voltage_no_source))
                - sp.diags(node_voltage_no_source)
                @ np.conj(electric_grid_model.node_admittance_matrix_no_source)
                @ A_matrix_inverse
                @ sp.identity(len(node_voltage_no_source))
            ).tocsc(),
        )
        self.sensitivity_voltage_by_power_wye_reactive[
            np.ix_(
                mesmo.utils.get_index(electric_grid_model.nodes, node_type="no_source"),
                mesmo.utils.get_index(electric_grid_model.nodes, node_type="no_source"),
            )
        ] = scipy.sparse.linalg.spsolve(
            B_matrix.tocsc(),
            (
                (1.0j * sp.identity(len(node_voltage_no_source)))
                - sp.diags(node_voltage_no_source)
                @ np.conj(electric_grid_model.node_admittance_matrix_no_source)
                @ A_matrix_inverse
                @ (-1.0j * sp.identity(len(node_voltage_no_source)))
            ).tocsc(),
        )
        # self.sensitivity_voltage_by_power_delta_active[np.ix_(
        #     mesmo.utils.get_index(electric_grid_model.nodes, node_type='no_source'),
        #     mesmo.utils.get_index(electric_grid_model.nodes, node_type='no_source')
        # )] = (
        #     ???
        # )
        # self.sensitivity_voltage_by_power_delta_reactive[np.ix_(
        #     mesmo.utils.get_index(electric_grid_model.nodes, node_type='no_source'),
        #     mesmo.utils.get_index(electric_grid_model.nodes, node_type='no_source')
        # )] = (
        #     ???
        # )

        self.sensitivity_voltage_by_der_power_active = (
            self.sensitivity_voltage_by_power_wye_active @ electric_grid_model.der_incidence_wye_matrix
            + self.sensitivity_voltage_by_power_delta_active @ electric_grid_model.der_incidence_delta_matrix
        )
        self.sensitivity_voltage_by_der_power_reactive = (
            self.sensitivity_voltage_by_power_wye_reactive @ electric_grid_model.der_incidence_wye_matrix
            + self.sensitivity_voltage_by_power_delta_reactive @ electric_grid_model.der_incidence_delta_matrix
        )

        self.sensitivity_voltage_magnitude_by_power_wye_active = sp.diags(
            abs(self.power_flow_solution.node_voltage_vector) ** -1
        ) @ np.real(
            sp.diags(np.conj(self.power_flow_solution.node_voltage_vector))
            @ self.sensitivity_voltage_by_power_wye_active
        )
        self.sensitivity_voltage_magnitude_by_power_wye_reactive = sp.diags(
            abs(self.power_flow_solution.node_voltage_vector) ** -1
        ) @ np.real(
            sp.diags(np.conj(self.power_flow_solution.node_voltage_vector))
            @ self.sensitivity_voltage_by_power_wye_reactive
        )
        self.sensitivity_voltage_magnitude_by_power_delta_active = sp.diags(
            abs(self.power_flow_solution.node_voltage_vector) ** -1
        ) @ np.real(
            sp.diags(np.conj(self.power_flow_solution.node_voltage_vector))
            @ self.sensitivity_voltage_by_power_delta_active
        )
        self.sensitivity_voltage_magnitude_by_power_delta_reactive = sp.diags(
            abs(self.power_flow_solution.node_voltage_vector) ** -1
        ) @ np.real(
            sp.diags(np.conj(self.power_flow_solution.node_voltage_vector))
            @ self.sensitivity_voltage_by_power_delta_reactive
        )

        self.sensitivity_voltage_magnitude_by_der_power_active = (
            self.sensitivity_voltage_magnitude_by_power_wye_active @ electric_grid_model.der_incidence_wye_matrix
            + self.sensitivity_voltage_magnitude_by_power_delta_active @ electric_grid_model.der_incidence_delta_matrix
        )
        self.sensitivity_voltage_magnitude_by_der_power_reactive = (
            self.sensitivity_voltage_magnitude_by_power_wye_reactive @ electric_grid_model.der_incidence_wye_matrix
            + self.sensitivity_voltage_magnitude_by_power_delta_reactive
            @ electric_grid_model.der_incidence_delta_matrix
        )

        # Calculate branch power sensitivity matrices.
        # TODO: Check for issues with reactive branch power component.
        sensitivity_branch_power_1_by_voltage_active = (
            sp.diags(
                np.conj(
                    self.electric_grid_model.branch_incidence_1_matrix
                    @ np.transpose([self.power_flow_solution.node_voltage_vector])
                ).ravel()
            )
            @ self.electric_grid_model.branch_admittance_1_matrix
        )
        sensitivity_branch_power_1_by_voltage_reactive = (
            sp.diags(
                -1.0
                * np.conj(
                    self.electric_grid_model.branch_incidence_1_matrix
                    @ np.transpose([self.power_flow_solution.node_voltage_vector])
                ).ravel()
            )
            @ self.electric_grid_model.branch_admittance_1_matrix
        )
        sensitivity_branch_power_2_by_voltage_active = (
            sp.diags(
                np.conj(
                    self.electric_grid_model.branch_incidence_2_matrix
                    @ np.transpose([self.power_flow_solution.node_voltage_vector])
                ).ravel()
            )
            @ self.electric_grid_model.branch_admittance_2_matrix
        )
        sensitivity_branch_power_2_by_voltage_reactive = (
            sp.diags(
                -1.0
                * np.conj(
                    self.electric_grid_model.branch_incidence_2_matrix
                    @ np.transpose([self.power_flow_solution.node_voltage_vector])
                ).ravel()
            )
            @ self.electric_grid_model.branch_admittance_2_matrix
        )

        self.sensitivity_branch_power_1_by_power_wye_active = np.conj(
            sensitivity_branch_power_1_by_voltage_active @ self.sensitivity_voltage_by_power_wye_active
        )
        self.sensitivity_branch_power_1_by_power_wye_reactive = -1.0 * np.conj(
            sensitivity_branch_power_1_by_voltage_reactive @ self.sensitivity_voltage_by_power_wye_reactive
        )
        self.sensitivity_branch_power_1_by_power_delta_active = np.conj(
            sensitivity_branch_power_1_by_voltage_active @ self.sensitivity_voltage_by_power_delta_active
        )
        self.sensitivity_branch_power_1_by_power_delta_reactive = -1.0 * np.conj(
            sensitivity_branch_power_1_by_voltage_reactive @ self.sensitivity_voltage_by_power_delta_reactive
        )
        self.sensitivity_branch_power_2_by_power_wye_active = np.conj(
            sensitivity_branch_power_2_by_voltage_active @ self.sensitivity_voltage_by_power_wye_active
        )
        self.sensitivity_branch_power_2_by_power_wye_reactive = -1.0 * np.conj(
            sensitivity_branch_power_2_by_voltage_reactive @ self.sensitivity_voltage_by_power_wye_reactive
        )
        self.sensitivity_branch_power_2_by_power_delta_active = np.conj(
            sensitivity_branch_power_2_by_voltage_active @ self.sensitivity_voltage_by_power_delta_active
        )
        self.sensitivity_branch_power_2_by_power_delta_reactive = -1.0 * np.conj(
            sensitivity_branch_power_2_by_voltage_reactive @ self.sensitivity_voltage_by_power_delta_reactive
        )

        self.sensitivity_branch_power_1_by_der_power_active = (
            self.sensitivity_branch_power_1_by_power_wye_active @ electric_grid_model.der_incidence_wye_matrix
            + self.sensitivity_branch_power_1_by_power_delta_active @ electric_grid_model.der_incidence_delta_matrix
        )
        self.sensitivity_branch_power_1_by_der_power_reactive = (
            self.sensitivity_branch_power_1_by_power_wye_reactive @ electric_grid_model.der_incidence_wye_matrix
            + self.sensitivity_branch_power_1_by_power_delta_reactive @ electric_grid_model.der_incidence_delta_matrix
        )
        self.sensitivity_branch_power_2_by_der_power_active = (
            self.sensitivity_branch_power_2_by_power_wye_active @ electric_grid_model.der_incidence_wye_matrix
            + self.sensitivity_branch_power_2_by_power_delta_active @ electric_grid_model.der_incidence_delta_matrix
        )
        self.sensitivity_branch_power_2_by_der_power_reactive = (
            self.sensitivity_branch_power_2_by_power_wye_reactive @ electric_grid_model.der_incidence_wye_matrix
            + self.sensitivity_branch_power_2_by_power_delta_reactive @ electric_grid_model.der_incidence_delta_matrix
        )

        self.sensitivity_branch_power_1_magnitude_by_power_wye_active = sp.diags(
            abs(self.power_flow_solution.branch_power_vector_1) ** -1
        ) @ np.real(
            sp.diags(np.conj(self.power_flow_solution.branch_power_vector_1))
            @ self.sensitivity_branch_power_1_by_power_wye_active
        )
        self.sensitivity_branch_power_1_magnitude_by_power_wye_reactive = sp.diags(
            abs(self.power_flow_solution.branch_power_vector_1) ** -1
        ) @ np.real(
            sp.diags(np.conj(self.power_flow_solution.branch_power_vector_1))
            @ self.sensitivity_branch_power_1_by_power_wye_reactive
        )
        self.sensitivity_branch_power_1_magnitude_by_power_delta_active = sp.diags(
            abs(self.power_flow_solution.branch_power_vector_1) ** -1
        ) @ np.real(
            sp.diags(np.conj(self.power_flow_solution.branch_power_vector_1))
            @ self.sensitivity_branch_power_1_by_power_delta_active
        )
        self.sensitivity_branch_power_1_magnitude_by_power_delta_reactive = sp.diags(
            abs(self.power_flow_solution.branch_power_vector_1) ** -1
        ) @ np.real(
            sp.diags(np.conj(self.power_flow_solution.branch_power_vector_1))
            @ self.sensitivity_branch_power_1_by_power_delta_reactive
        )
        self.sensitivity_branch_power_2_magnitude_by_power_wye_active = sp.diags(
            abs(self.power_flow_solution.branch_power_vector_2) ** -1
        ) @ np.real(
            sp.diags(np.conj(self.power_flow_solution.branch_power_vector_2))
            @ self.sensitivity_branch_power_2_by_power_wye_active
        )
        self.sensitivity_branch_power_2_magnitude_by_power_wye_reactive = sp.diags(
            abs(self.power_flow_solution.branch_power_vector_2) ** -1
        ) @ np.real(
            sp.diags(np.conj(self.power_flow_solution.branch_power_vector_2))
            @ self.sensitivity_branch_power_2_by_power_wye_reactive
        )
        self.sensitivity_branch_power_2_magnitude_by_power_delta_active = sp.diags(
            abs(self.power_flow_solution.branch_power_vector_2) ** -1
        ) @ np.real(
            sp.diags(np.conj(self.power_flow_solution.branch_power_vector_2))
            @ self.sensitivity_branch_power_2_by_power_delta_active
        )
        self.sensitivity_branch_power_2_magnitude_by_power_delta_reactive = sp.diags(
            abs(self.power_flow_solution.branch_power_vector_2) ** -1
        ) @ np.real(
            sp.diags(np.conj(self.power_flow_solution.branch_power_vector_2))
            @ self.sensitivity_branch_power_2_by_power_delta_reactive
        )

        self.sensitivity_branch_power_1_magnitude_by_der_power_active = (
            self.sensitivity_branch_power_1_magnitude_by_power_wye_active @ electric_grid_model.der_incidence_wye_matrix
            + self.sensitivity_branch_power_1_magnitude_by_power_delta_active
            @ electric_grid_model.der_incidence_delta_matrix
        )
        self.sensitivity_branch_power_1_magnitude_by_der_power_reactive = (
            self.sensitivity_branch_power_1_magnitude_by_power_wye_reactive
            @ electric_grid_model.der_incidence_wye_matrix
            + self.sensitivity_branch_power_1_magnitude_by_power_delta_reactive
            @ electric_grid_model.der_incidence_delta_matrix
        )
        self.sensitivity_branch_power_2_magnitude_by_der_power_active = (
            self.sensitivity_branch_power_2_magnitude_by_power_wye_active @ electric_grid_model.der_incidence_wye_matrix
            + self.sensitivity_branch_power_2_magnitude_by_power_delta_active
            @ electric_grid_model.der_incidence_delta_matrix
        )
        self.sensitivity_branch_power_2_magnitude_by_der_power_reactive = (
            self.sensitivity_branch_power_2_magnitude_by_power_wye_reactive
            @ electric_grid_model.der_incidence_wye_matrix
            + self.sensitivity_branch_power_2_magnitude_by_power_delta_reactive
            @ electric_grid_model.der_incidence_delta_matrix
        )

        # Calculate loss sensitivity matrices.
        self.sensitivity_loss_active_by_power_wye_active = sum(
            np.real(
                self.sensitivity_branch_power_1_by_power_wye_active
                + self.sensitivity_branch_power_2_by_power_wye_active
            )
        )
        self.sensitivity_loss_active_by_power_wye_reactive = sum(
            np.real(
                self.sensitivity_branch_power_1_by_power_wye_reactive
                + self.sensitivity_branch_power_2_by_power_wye_reactive
            )
        )
        self.sensitivity_loss_active_by_power_delta_active = sum(
            np.real(
                self.sensitivity_branch_power_1_by_power_delta_active
                + self.sensitivity_branch_power_2_by_power_delta_active
            )
        )
        self.sensitivity_loss_active_by_power_delta_reactive = sum(
            np.real(
                self.sensitivity_branch_power_1_by_power_delta_reactive
                + self.sensitivity_branch_power_2_by_power_delta_reactive
            )
        )
        self.sensitivity_loss_reactive_by_power_wye_active = sum(
            np.imag(
                self.sensitivity_branch_power_1_by_power_wye_active
                + self.sensitivity_branch_power_2_by_power_wye_active
            )
        )
        self.sensitivity_loss_reactive_by_power_wye_reactive = sum(
            np.imag(
                self.sensitivity_branch_power_1_by_power_wye_reactive
                + self.sensitivity_branch_power_2_by_power_wye_reactive
            )
        )
        self.sensitivity_loss_reactive_by_power_delta_active = sum(
            np.imag(
                self.sensitivity_branch_power_1_by_power_delta_active
                + self.sensitivity_branch_power_2_by_power_delta_active
            )
        )
        self.sensitivity_loss_reactive_by_power_delta_reactive = sum(
            np.imag(
                self.sensitivity_branch_power_1_by_power_delta_reactive
                + self.sensitivity_branch_power_2_by_power_delta_reactive
            )
        )

        self.sensitivity_loss_active_by_der_power_active = (
            self.sensitivity_loss_active_by_power_wye_active @ electric_grid_model.der_incidence_wye_matrix
            + self.sensitivity_loss_active_by_power_delta_active @ electric_grid_model.der_incidence_delta_matrix
        )
        self.sensitivity_loss_active_by_der_power_reactive = (
            self.sensitivity_loss_active_by_power_wye_reactive @ electric_grid_model.der_incidence_wye_matrix
            + self.sensitivity_loss_active_by_power_delta_reactive @ electric_grid_model.der_incidence_delta_matrix
        )
        self.sensitivity_loss_reactive_by_der_power_active = (
            self.sensitivity_loss_reactive_by_power_wye_active @ electric_grid_model.der_incidence_wye_matrix
            + self.sensitivity_loss_reactive_by_power_delta_active @ electric_grid_model.der_incidence_delta_matrix
        )
        self.sensitivity_loss_reactive_by_der_power_reactive = (
            self.sensitivity_loss_reactive_by_power_wye_reactive @ electric_grid_model.der_incidence_wye_matrix
            + self.sensitivity_loss_reactive_by_power_delta_reactive @ electric_grid_model.der_incidence_delta_matrix
        )


class LinearElectricGridModelSet(mesmo.utils.ObjectBase):

    linear_electric_grid_models: typing.Dict[pd.Timestamp, LinearElectricGridModelBase]
    electric_grid_model: ElectricGridModel
    timesteps: pd.Index

    @multimethod.multimethod
    def __init__(self, scenario_name: str):

        # Obtain electric grid model & reference power flow solution.
        electric_grid_model = ElectricGridModel(scenario_name)
        power_flow_solution = PowerFlowSolutionFixedPoint(electric_grid_model)

        self.__init__(electric_grid_model, power_flow_solution)

    @multimethod.multimethod
    def __init__(
        self,
        electric_grid_model: ElectricGridModel,
        power_flow_solution: PowerFlowSolutionBase,
        linear_electric_grid_model_method: typing.Type[LinearElectricGridModelBase] = LinearElectricGridModelGlobal,
    ):

        self.check_linear_electric_grid_model_method(linear_electric_grid_model_method)

        # Obtain linear electric grid models.
        linear_electric_grid_model = linear_electric_grid_model_method(electric_grid_model, power_flow_solution)
        linear_electric_grid_models = dict(
            zip(electric_grid_model.timesteps, itertools.repeat(linear_electric_grid_model))
        )

        self.__init__(electric_grid_model, linear_electric_grid_models)

    @multimethod.multimethod
    def __init__(
        self,
        electric_grid_model: ElectricGridModel,
        power_flow_solution_set: PowerFlowSolutionSet,
        linear_electric_grid_model_method: typing.Type[LinearElectricGridModelBase] = LinearElectricGridModelLocal,
    ):

        self.check_linear_electric_grid_model_method(linear_electric_grid_model_method)

        # Obtain linear electric grid models.
        linear_electric_grid_models = mesmo.utils.starmap(
            linear_electric_grid_model_method,
            zip(itertools.repeat(electric_grid_model), power_flow_solution_set.power_flow_solutions.values()),
        )
        linear_electric_grid_models = dict(zip(electric_grid_model.timesteps, linear_electric_grid_models))

        self.__init__(electric_grid_model, linear_electric_grid_models)

    @multimethod.multimethod
    def __init__(
        self,
        electric_grid_model: ElectricGridModel,
        linear_electric_grid_models: typing.Dict[pd.Timestamp, LinearElectricGridModelBase],
    ):

        # Store attributes.
        self.electric_grid_model = electric_grid_model
        self.timesteps = self.electric_grid_model.timesteps
        self.linear_electric_grid_models = linear_electric_grid_models

    # Define `update()` as alternative entry point for `__init__()`
    update = __init__

    @staticmethod
    def check_linear_electric_grid_model_method(linear_electric_grid_model_method):

        if not issubclass(linear_electric_grid_model_method, LinearElectricGridModelBase):
            raise ValueError(f"Invalid linear electric grid model method: {linear_electric_grid_model_method}")

    def define_optimization_problem(
        self,
        optimization_problem: mesmo.solutions.OptimizationProblem,
        price_data: mesmo.data_interface.PriceData,
        scenarios: typing.Union[list, pd.Index] = None,
        **kwargs,
    ):

        # Defined optimization problem definitions through respective sub-methods.
        self.define_optimization_variables(optimization_problem, scenarios=scenarios)
        self.define_optimization_parameters(optimization_problem, price_data, scenarios=scenarios, **kwargs)
        self.define_optimization_constraints(optimization_problem, scenarios=scenarios)
        self.define_optimization_objective(optimization_problem, scenarios=scenarios)

    def define_optimization_variables(
        self, optimization_problem: mesmo.solutions.OptimizationProblem, scenarios: typing.Union[list, pd.Index] = None
    ):

        # If no scenarios given, obtain default value.
        if scenarios is None:
            scenarios = [None]

        # Define DER power vector variables.
        optimization_problem.define_variable(
            "der_active_power_vector", scenario=scenarios, timestep=self.timesteps, der=self.electric_grid_model.ders
        )
        optimization_problem.define_variable(
            "der_reactive_power_vector", scenario=scenarios, timestep=self.timesteps, der=self.electric_grid_model.ders
        )

        # Define node voltage magnitude variable.
        optimization_problem.define_variable(
            "node_voltage_magnitude_vector",
            scenario=scenarios,
            timestep=self.timesteps,
            node=self.electric_grid_model.nodes,
        )

        # Define branch power magnitude variables.
        optimization_problem.define_variable(
            "branch_power_magnitude_vector_1",
            scenario=scenarios,
            timestep=self.timesteps,
            branch=self.electric_grid_model.branches,
        )
        optimization_problem.define_variable(
            "branch_power_magnitude_vector_2",
            scenario=scenarios,
            timestep=self.timesteps,
            branch=self.electric_grid_model.branches,
        )

        # Define loss variables.
        optimization_problem.define_variable("loss_active", scenario=scenarios, timestep=self.timesteps)
        optimization_problem.define_variable("loss_reactive", scenario=scenarios, timestep=self.timesteps)

    def define_optimization_parameters(
        self,
        optimization_problem: mesmo.solutions.OptimizationProblem,
        price_data: mesmo.data_interface.PriceData,
        node_voltage_magnitude_vector_minimum: np.ndarray = None,
        node_voltage_magnitude_vector_maximum: np.ndarray = None,
        branch_power_magnitude_vector_maximum: np.ndarray = None,
        scenarios: typing.Union[list, pd.Index] = None,
    ):

        # If no scenarios given, obtain default value.
        if scenarios is None:
            scenarios = [None]

        # Obtain timestep interval in hours, for conversion of power to energy.
        timestep_interval_hours = (self.timesteps[1] - self.timesteps[0]) / pd.Timedelta("1h")

        # Define voltage variable terms.
        optimization_problem.define_parameter(
            "voltage_active_term",
            sp.block_diag(
                [
                    sp.diags(np.abs(linear_electric_grid_model.electric_grid_model.node_voltage_vector_reference) ** -1)
                    @ linear_electric_grid_model.sensitivity_voltage_magnitude_by_der_power_active
                    @ sp.diags(np.real(linear_electric_grid_model.electric_grid_model.der_power_vector_reference))
                    for linear_electric_grid_model in self.linear_electric_grid_models.values()
                ]
            ),
        )
        optimization_problem.define_parameter(
            "voltage_reactive_term",
            sp.block_diag(
                [
                    sp.diags(np.abs(linear_electric_grid_model.electric_grid_model.node_voltage_vector_reference) ** -1)
                    @ linear_electric_grid_model.sensitivity_voltage_magnitude_by_der_power_reactive
                    @ sp.diags(np.imag(linear_electric_grid_model.electric_grid_model.der_power_vector_reference))
                    for linear_electric_grid_model in self.linear_electric_grid_models.values()
                ]
            ),
        )

        # Define voltage constant term.
        optimization_problem.define_parameter(
            "voltage_constant",
            np.concatenate(
                [
                    sp.diags(np.abs(linear_electric_grid_model.electric_grid_model.node_voltage_vector_reference) ** -1)
                    @ (
                        np.transpose([np.abs(linear_electric_grid_model.power_flow_solution.node_voltage_vector)])
                        - linear_electric_grid_model.sensitivity_voltage_magnitude_by_der_power_active
                        @ np.transpose([np.real(linear_electric_grid_model.power_flow_solution.der_power_vector)])
                        - linear_electric_grid_model.sensitivity_voltage_magnitude_by_der_power_reactive
                        @ np.transpose([np.imag(linear_electric_grid_model.power_flow_solution.der_power_vector)])
                    )
                    for linear_electric_grid_model in self.linear_electric_grid_models.values()
                ]
            ),
        )

        # Define branch flow (direction 1) variable terms.
        optimization_problem.define_parameter(
            "branch_power_1_active_term",
            sp.block_diag(
                [
                    sp.diags(
                        linear_electric_grid_model.electric_grid_model.branch_power_vector_magnitude_reference**-1
                    )
                    @ linear_electric_grid_model.sensitivity_branch_power_1_magnitude_by_der_power_active
                    @ sp.diags(np.real(linear_electric_grid_model.electric_grid_model.der_power_vector_reference))
                    for linear_electric_grid_model in self.linear_electric_grid_models.values()
                ]
            ),
        )
        optimization_problem.define_parameter(
            "branch_power_1_reactive_term",
            sp.block_diag(
                [
                    sp.diags(
                        linear_electric_grid_model.electric_grid_model.branch_power_vector_magnitude_reference**-1
                    )
                    @ linear_electric_grid_model.sensitivity_branch_power_1_magnitude_by_der_power_reactive
                    @ sp.diags(np.imag(linear_electric_grid_model.electric_grid_model.der_power_vector_reference))
                    for linear_electric_grid_model in self.linear_electric_grid_models.values()
                ]
            ),
        )

        # Define branch flow (direction 1) constant terms.
        optimization_problem.define_parameter(
            "branch_power_1_constant",
            np.concatenate(
                [
                    sp.diags(
                        linear_electric_grid_model.electric_grid_model.branch_power_vector_magnitude_reference**-1
                    )
                    @ (
                        np.transpose([np.abs(linear_electric_grid_model.power_flow_solution.branch_power_vector_1)])
                        - linear_electric_grid_model.sensitivity_branch_power_1_magnitude_by_der_power_active
                        @ np.transpose([np.real(linear_electric_grid_model.power_flow_solution.der_power_vector)])
                        - linear_electric_grid_model.sensitivity_branch_power_1_magnitude_by_der_power_reactive
                        @ np.transpose([np.imag(linear_electric_grid_model.power_flow_solution.der_power_vector)])
                    )
                    for linear_electric_grid_model in self.linear_electric_grid_models.values()
                ]
            ),
        )

        # Define branch flow (direction 2) variable terms.
        optimization_problem.define_parameter(
            "branch_power_2_active_term",
            sp.block_diag(
                [
                    sp.diags(
                        linear_electric_grid_model.electric_grid_model.branch_power_vector_magnitude_reference**-1
                    )
                    @ linear_electric_grid_model.sensitivity_branch_power_2_magnitude_by_der_power_active
                    @ sp.diags(np.real(linear_electric_grid_model.electric_grid_model.der_power_vector_reference))
                    for linear_electric_grid_model in self.linear_electric_grid_models.values()
                ]
            ),
        )
        optimization_problem.define_parameter(
            "branch_power_2_reactive_term",
            sp.block_diag(
                [
                    sp.diags(
                        linear_electric_grid_model.electric_grid_model.branch_power_vector_magnitude_reference**-1
                    )
                    @ linear_electric_grid_model.sensitivity_branch_power_2_magnitude_by_der_power_reactive
                    @ sp.diags(np.imag(linear_electric_grid_model.electric_grid_model.der_power_vector_reference))
                    for linear_electric_grid_model in self.linear_electric_grid_models.values()
                ]
            ),
        )

        # Define branch flow (direction 2) constant term.
        optimization_problem.define_parameter(
            "branch_power_2_constant",
            np.concatenate(
                [
                    sp.diags(
                        linear_electric_grid_model.electric_grid_model.branch_power_vector_magnitude_reference**-1
                    )
                    @ (
                        np.transpose([np.abs(linear_electric_grid_model.power_flow_solution.branch_power_vector_2)])
                        - linear_electric_grid_model.sensitivity_branch_power_2_magnitude_by_der_power_active
                        @ np.transpose([np.real(linear_electric_grid_model.power_flow_solution.der_power_vector)])
                        - linear_electric_grid_model.sensitivity_branch_power_2_magnitude_by_der_power_reactive
                        @ np.transpose([np.imag(linear_electric_grid_model.power_flow_solution.der_power_vector)])
                    )
                    for linear_electric_grid_model in self.linear_electric_grid_models.values()
                ]
            ),
        )

        # Define active loss variable terms.
        optimization_problem.define_parameter(
            "loss_active_active_term",
            sp.block_diag(
                [
                    linear_electric_grid_model.sensitivity_loss_active_by_der_power_active
                    @ sp.diags(np.real(linear_electric_grid_model.electric_grid_model.der_power_vector_reference))
                    for linear_electric_grid_model in self.linear_electric_grid_models.values()
                ]
            ),
        )
        optimization_problem.define_parameter(
            "loss_active_reactive_term",
            sp.block_diag(
                [
                    linear_electric_grid_model.sensitivity_loss_active_by_der_power_reactive
                    @ sp.diags(np.imag(linear_electric_grid_model.electric_grid_model.der_power_vector_reference))
                    for linear_electric_grid_model in self.linear_electric_grid_models.values()
                ]
            ),
        )

        # Define active loss constant term.
        optimization_problem.define_parameter(
            "loss_active_constant",
            np.concatenate(
                [
                    np.real(linear_electric_grid_model.power_flow_solution.loss)
                    - linear_electric_grid_model.sensitivity_loss_active_by_der_power_active
                    @ np.transpose([np.real(linear_electric_grid_model.power_flow_solution.der_power_vector)])
                    - linear_electric_grid_model.sensitivity_loss_active_by_der_power_reactive
                    @ np.transpose([np.imag(linear_electric_grid_model.power_flow_solution.der_power_vector)])
                    for linear_electric_grid_model in self.linear_electric_grid_models.values()
                ]
            ),
        )

        # Define reactive loss variable terms.
        optimization_problem.define_parameter(
            "loss_reactive_active_term",
            sp.block_diag(
                [
                    linear_electric_grid_model.sensitivity_loss_reactive_by_der_power_active
                    @ sp.diags(np.real(linear_electric_grid_model.electric_grid_model.der_power_vector_reference))
                    for linear_electric_grid_model in self.linear_electric_grid_models.values()
                ]
            ),
        )
        optimization_problem.define_parameter(
            "loss_reactive_reactive_term",
            sp.block_diag(
                [
                    linear_electric_grid_model.sensitivity_loss_reactive_by_der_power_reactive
                    @ sp.diags(np.imag(linear_electric_grid_model.electric_grid_model.der_power_vector_reference))
                    for linear_electric_grid_model in self.linear_electric_grid_models.values()
                ]
            ),
        )

        # Define active loss constant term.
        optimization_problem.define_parameter(
            "loss_reactive_constant",
            np.concatenate(
                [
                    np.imag(linear_electric_grid_model.power_flow_solution.loss)
                    - linear_electric_grid_model.sensitivity_loss_reactive_by_der_power_active
                    @ np.transpose([np.real(linear_electric_grid_model.power_flow_solution.der_power_vector)])
                    - linear_electric_grid_model.sensitivity_loss_reactive_by_der_power_reactive
                    @ np.transpose([np.imag(linear_electric_grid_model.power_flow_solution.der_power_vector)])
                    for linear_electric_grid_model in self.linear_electric_grid_models.values()
                ]
            ),
        )

        # Define voltage limits.
        optimization_problem.define_parameter(
            "voltage_limit_minimum",
            np.concatenate(
                [
                    node_voltage_magnitude_vector_minimum.ravel()
                    / np.abs(linear_electric_grid_model.electric_grid_model.node_voltage_vector_reference)
                    for linear_electric_grid_model in self.linear_electric_grid_models.values()
                ]
            )
            if node_voltage_magnitude_vector_minimum is not None
            else -np.inf * np.ones((len(self.electric_grid_model.nodes) * len(self.timesteps),)),
        )
        optimization_problem.define_parameter(
            "voltage_limit_maximum",
            np.concatenate(
                [
                    node_voltage_magnitude_vector_maximum.ravel()
                    / np.abs(linear_electric_grid_model.electric_grid_model.node_voltage_vector_reference)
                    for linear_electric_grid_model in self.linear_electric_grid_models.values()
                ]
            )
            if node_voltage_magnitude_vector_maximum is not None
            else +np.inf * np.ones((len(self.electric_grid_model.nodes) * len(self.timesteps),)),
        )

        # Define branch flow limits.
        optimization_problem.define_parameter(
            "branch_power_minimum",
            np.concatenate(
                [
                    -branch_power_magnitude_vector_maximum.ravel()
                    / linear_electric_grid_model.electric_grid_model.branch_power_vector_magnitude_reference
                    for linear_electric_grid_model in self.linear_electric_grid_models.values()
                ]
            )
            if branch_power_magnitude_vector_maximum is not None
            else -np.inf * np.ones((len(self.electric_grid_model.branches) * len(self.timesteps),)),
        )
        optimization_problem.define_parameter(
            "branch_power_maximum",
            np.concatenate(
                [
                    branch_power_magnitude_vector_maximum.ravel()
                    / linear_electric_grid_model.electric_grid_model.branch_power_vector_magnitude_reference
                    for linear_electric_grid_model in self.linear_electric_grid_models.values()
                ]
            )
            if branch_power_magnitude_vector_maximum is not None
            else +np.inf * np.ones((len(self.electric_grid_model.branches) * len(self.timesteps),)),
        )

        # Define objective parameters.
        optimization_problem.define_parameter(
            "electric_grid_active_power_cost",
            np.array([price_data.price_timeseries.loc[:, ("active_power", "source", "source")].values])
            * -1.0
            * timestep_interval_hours  # In Wh.
            @ sp.block_diag(
                [np.array([np.real(self.electric_grid_model.der_power_vector_reference)])] * len(self.timesteps)
            ),
        )
        optimization_problem.define_parameter(
            "electric_grid_active_power_cost_sensitivity",
            price_data.price_sensitivity_coefficient
            * timestep_interval_hours  # In Wh.
            * np.concatenate([np.real(self.electric_grid_model.der_power_vector_reference) ** 2] * len(self.timesteps)),
        )
        optimization_problem.define_parameter(
            "electric_grid_reactive_power_cost",
            np.array([price_data.price_timeseries.loc[:, ("reactive_power", "source", "source")].values])
            * -1.0
            * timestep_interval_hours  # In Wh.
            @ sp.block_diag(
                [np.array([np.imag(self.electric_grid_model.der_power_vector_reference)])] * len(self.timesteps)
            ),
        )
        optimization_problem.define_parameter(
            "electric_grid_reactive_power_cost_sensitivity",
            price_data.price_sensitivity_coefficient
            * timestep_interval_hours  # In Wh.
            * np.concatenate([np.imag(self.electric_grid_model.der_power_vector_reference) ** 2] * len(self.timesteps)),
        )
        optimization_problem.define_parameter(
            "electric_grid_loss_active_cost",
            price_data.price_timeseries.loc[:, ("active_power", "source", "source")].values
            * timestep_interval_hours,  # In Wh.
        )
        optimization_problem.define_parameter(
            "electric_grid_loss_active_cost_sensitivity",
            price_data.price_sensitivity_coefficient * timestep_interval_hours,  # In Wh.
        )

    def define_optimization_constraints(
        self, optimization_problem: mesmo.solutions.OptimizationProblem, scenarios: typing.Union[list, pd.Index] = None
    ):

        # If no scenarios given, obtain default value.
        if scenarios is None:
            scenarios = [None]

        # Define voltage equation.
        optimization_problem.define_constraint(
            (
                "variable",
                1.0,
                dict(
                    name="node_voltage_magnitude_vector",
                    scenario=scenarios,
                    timestep=self.timesteps,
                    node=self.electric_grid_model.nodes,
                ),
            ),
            "==",
            (
                "variable",
                "voltage_active_term",
                dict(
                    name="der_active_power_vector",
                    scenario=scenarios,
                    timestep=self.timesteps,
                    der=self.electric_grid_model.ders,
                ),
            ),
            (
                "variable",
                "voltage_reactive_term",
                dict(
                    name="der_reactive_power_vector",
                    scenario=scenarios,
                    timestep=self.timesteps,
                    der=self.electric_grid_model.ders,
                ),
            ),
            ("constant", "voltage_constant", dict(scenario=scenarios, timestep=self.timesteps)),
            broadcast="scenario",
        )

        # Define branch flow (direction 1) equation.
        optimization_problem.define_constraint(
            (
                "variable",
                1.0,
                dict(
                    name="branch_power_magnitude_vector_1",
                    scenario=scenarios,
                    timestep=self.timesteps,
                    branch=self.electric_grid_model.branches,
                ),
            ),
            "==",
            (
                "variable",
                "branch_power_1_active_term",
                dict(
                    name="der_active_power_vector",
                    scenario=scenarios,
                    timestep=self.timesteps,
                    der=self.electric_grid_model.ders,
                ),
            ),
            (
                "variable",
                "branch_power_1_reactive_term",
                dict(
                    name="der_reactive_power_vector",
                    scenario=scenarios,
                    timestep=self.timesteps,
                    der=self.electric_grid_model.ders,
                ),
            ),
            ("constant", "branch_power_1_constant", dict(scenario=scenarios, timestep=self.timesteps)),
            broadcast="scenario",
        )

        # Define branch flow (direction 2) equation.
        optimization_problem.define_constraint(
            (
                "variable",
                1.0,
                dict(
                    name="branch_power_magnitude_vector_2",
                    scenario=scenarios,
                    timestep=self.timesteps,
                    branch=self.electric_grid_model.branches,
                ),
            ),
            "==",
            (
                "variable",
                "branch_power_2_active_term",
                dict(
                    name="der_active_power_vector",
                    scenario=scenarios,
                    timestep=self.timesteps,
                    der=self.electric_grid_model.ders,
                ),
            ),
            (
                "variable",
                "branch_power_2_reactive_term",
                dict(
                    name="der_reactive_power_vector",
                    scenario=scenarios,
                    timestep=self.timesteps,
                    der=self.electric_grid_model.ders,
                ),
            ),
            ("constant", "branch_power_2_constant", dict(scenario=scenarios, timestep=self.timesteps)),
            broadcast="scenario",
        )

        # Define active loss equation.
        optimization_problem.define_constraint(
            ("variable", 1.0, dict(name="loss_active", scenario=scenarios, timestep=self.timesteps)),
            "==",
            (
                "variable",
                "loss_active_active_term",
                dict(
                    name="der_active_power_vector",
                    scenario=scenarios,
                    timestep=self.timesteps,
                    der=self.electric_grid_model.ders,
                ),
            ),
            (
                "variable",
                "loss_active_reactive_term",
                dict(
                    name="der_reactive_power_vector",
                    scenario=scenarios,
                    timestep=self.timesteps,
                    der=self.electric_grid_model.ders,
                ),
            ),
            ("constant", "loss_active_constant", dict(scenario=scenarios, timestep=self.timesteps)),
            broadcast="scenario",
        )

        # Define reactive loss equation.
        optimization_problem.define_constraint(
            ("variable", 1.0, dict(name="loss_reactive", scenario=scenarios, timestep=self.timesteps)),
            "==",
            (
                "variable",
                "loss_reactive_active_term",
                dict(
                    name="der_active_power_vector",
                    scenario=scenarios,
                    timestep=self.timesteps,
                    der=self.electric_grid_model.ders,
                ),
            ),
            (
                "variable",
                "loss_reactive_reactive_term",
                dict(
                    name="der_reactive_power_vector",
                    scenario=scenarios,
                    timestep=self.timesteps,
                    der=self.electric_grid_model.ders,
                ),
            ),
            ("constant", "loss_reactive_constant", dict(scenario=scenarios, timestep=self.timesteps)),
            broadcast="scenario",
        )

        # Define voltage limits.
        # Add dedicated keys to enable retrieving dual variables.
        optimization_problem.define_constraint(
            (
                "variable",
                1.0,
                dict(
                    name="node_voltage_magnitude_vector",
                    scenario=scenarios,
                    timestep=self.timesteps,
                    node=self.electric_grid_model.nodes,
                ),
            ),
            ">=",
            ("constant", "voltage_limit_minimum", dict(scenario=scenarios, timestep=self.timesteps)),
            keys=dict(
                name="voltage_magnitude_vector_minimum_constraint",
                scenario=scenarios,
                timestep=self.timesteps,
                node=self.electric_grid_model.nodes,
            ),
            broadcast="scenario",
        )
        optimization_problem.define_constraint(
            (
                "variable",
                1.0,
                dict(
                    name="node_voltage_magnitude_vector",
                    scenario=scenarios,
                    timestep=self.timesteps,
                    node=self.electric_grid_model.nodes,
                ),
            ),
            "<=",
            ("constant", "voltage_limit_maximum", dict(scenario=scenarios, timestep=self.timesteps)),
            keys=dict(
                name="voltage_magnitude_vector_maximum_constraint",
                scenario=scenarios,
                timestep=self.timesteps,
                node=self.electric_grid_model.nodes,
            ),
            broadcast="scenario",
        )

        # Define branch flow limits.
        # Add dedicated keys to enable retrieving dual variables.
        optimization_problem.define_constraint(
            (
                "variable",
                1.0,
                dict(
                    name="branch_power_magnitude_vector_1",
                    scenario=scenarios,
                    timestep=self.timesteps,
                    branch=self.electric_grid_model.branches,
                ),
            ),
            ">=",
            ("constant", "branch_power_minimum", dict(scenario=scenarios, timestep=self.timesteps)),
            keys=dict(
                name="branch_power_magnitude_vector_1_minimum_constraint",
                scenario=scenarios,
                timestep=self.timesteps,
                branch=self.electric_grid_model.branches,
            ),
            broadcast="scenario",
        )
        optimization_problem.define_constraint(
            (
                "variable",
                1.0,
                dict(
                    name="branch_power_magnitude_vector_1",
                    scenario=scenarios,
                    timestep=self.timesteps,
                    branch=self.electric_grid_model.branches,
                ),
            ),
            "<=",
            ("constant", "branch_power_maximum", dict(scenario=scenarios, timestep=self.timesteps)),
            keys=dict(
                name="branch_power_magnitude_vector_1_maximum_constraint",
                scenario=scenarios,
                timestep=self.timesteps,
                branch=self.electric_grid_model.branches,
            ),
            broadcast="scenario",
        )
        optimization_problem.define_constraint(
            (
                "variable",
                1.0,
                dict(
                    name="branch_power_magnitude_vector_2",
                    scenario=scenarios,
                    timestep=self.timesteps,
                    branch=self.electric_grid_model.branches,
                ),
            ),
            ">=",
            ("constant", "branch_power_minimum", dict(scenario=scenarios, timestep=self.timesteps)),
            keys=dict(
                name="branch_power_magnitude_vector_2_minimum_constraint",
                scenario=scenarios,
                timestep=self.timesteps,
                branch=self.electric_grid_model.branches,
            ),
            broadcast="scenario",
        )
        optimization_problem.define_constraint(
            (
                "variable",
                1.0,
                dict(
                    name="branch_power_magnitude_vector_2",
                    scenario=scenarios,
                    timestep=self.timesteps,
                    branch=self.electric_grid_model.branches,
                ),
            ),
            "<=",
            ("constant", "branch_power_maximum", dict(scenario=scenarios, timestep=self.timesteps)),
            keys=dict(
                name="branch_power_magnitude_vector_2_maximum_constraint",
                scenario=scenarios,
                timestep=self.timesteps,
                branch=self.electric_grid_model.branches,
            ),
            broadcast="scenario",
        )

    def define_optimization_objective(
        self, optimization_problem: mesmo.solutions.OptimizationProblem, scenarios: typing.Union[list, pd.Index] = None
    ):

        # If no scenarios given, obtain default value.
        if scenarios is None:
            scenarios = [None]

        # Set objective flag.
        optimization_problem.flags["has_electric_grid_objective"] = True

        # Define objective for electric loads.
        # - Defined as cost of electric supply at electric grid source node.
        # - Only defined here, if not yet defined as cost of electric power supply at the DER node
        #   in `mesmo.der_models.DERModel.define_optimization_objective`.
        if not optimization_problem.flags.get("has_der_objective"):

            # Active power cost / revenue.
            # - Cost for load / demand, revenue for generation / supply.
            optimization_problem.define_objective(
                (
                    "variable",
                    "electric_grid_active_power_cost",
                    dict(
                        name="der_active_power_vector",
                        scenario=scenarios,
                        timestep=self.timesteps,
                        der=self.electric_grid_model.ders,
                    ),
                ),
                (
                    "variable",
                    "electric_grid_active_power_cost_sensitivity",
                    dict(
                        name="der_active_power_vector",
                        scenario=scenarios,
                        timestep=self.timesteps,
                        der=self.electric_grid_model.ders,
                    ),
                    dict(
                        name="der_active_power_vector",
                        scenario=scenarios,
                        timestep=self.timesteps,
                        der=self.electric_grid_model.ders,
                    ),
                ),
                broadcast="scenario",
            )

            # Reactive power cost / revenue.
            # - Cost for load / demand, revenue for generation / supply.
            optimization_problem.define_objective(
                (
                    "variable",
                    "electric_grid_reactive_power_cost",
                    dict(
                        name="der_reactive_power_vector",
                        scenario=scenarios,
                        timestep=self.timesteps,
                        der=self.electric_grid_model.ders,
                    ),
                ),
                (
                    "variable",
                    "electric_grid_reactive_power_cost_sensitivity",
                    dict(
                        name="der_reactive_power_vector",
                        scenario=scenarios,
                        timestep=self.timesteps,
                        der=self.electric_grid_model.ders,
                    ),
                    dict(
                        name="der_reactive_power_vector",
                        scenario=scenarios,
                        timestep=self.timesteps,
                        der=self.electric_grid_model.ders,
                    ),
                ),
                broadcast="scenario",
            )

        # Define active loss cost.
        optimization_problem.define_objective(
            (
                "variable",
                "electric_grid_loss_active_cost",
                dict(name="loss_active", scenario=scenarios, timestep=self.timesteps),
            ),
            (
                "variable",
                "electric_grid_loss_active_cost_sensitivity",
                dict(name="loss_active", scenario=scenarios, timestep=self.timesteps),
                dict(name="loss_active", scenario=scenarios, timestep=self.timesteps),
            ),
            broadcast="scenario",
        )

    def evaluate_optimization_objective(
        self, results: ElectricGridOperationResults, price_data: mesmo.data_interface.PriceData
    ) -> float:

        # Instantiate optimization problem.
        optimization_problem = mesmo.solutions.OptimizationProblem()
        self.define_optimization_parameters(optimization_problem, price_data)
        self.define_optimization_variables(optimization_problem)
        self.define_optimization_objective(optimization_problem)

        # Instantiate variable vector.
        x_vector = np.zeros((len(optimization_problem.variables), 1))

        # Set variable vector values.
        objective_variable_names = [
            "der_active_power_vector_per_unit",
            "der_reactive_power_vector_per_unit",
            "loss_active",
        ]
        for variable_name in objective_variable_names:
            index = mesmo.utils.get_index(optimization_problem.variables, name=variable_name.replace("_per_unit", ""))
            x_vector[index, 0] = results[variable_name].values.ravel()

        # Obtain objective value.
        objective = optimization_problem.evaluate_objective(x_vector)

        return objective

    def get_optimization_dlmps(
        self,
        optimization_problem: mesmo.solutions.OptimizationProblem,
        price_data: mesmo.data_interface.PriceData,
        scenarios: typing.Union[list, pd.Index] = None,
    ) -> ElectricGridDLMPResults:

        # Obtain results index sets, depending on if / if not scenarios given.
        # TODO: Flatten index to align with other results.
        if scenarios in [None, [None]]:
            scenarios = [None]
            ders = self.electric_grid_model.ders
            nodes = self.electric_grid_model.nodes
            branches = self.electric_grid_model.branches
        else:
            ders = pd.MultiIndex.from_product(
                (scenarios, self.electric_grid_model.ders.to_flat_index()), names=["scenario", "der"]
            )
            nodes = pd.MultiIndex.from_product(
                (scenarios, self.electric_grid_model.nodes.to_flat_index()), names=["scenario", "node"]
            )
            branches = pd.MultiIndex.from_product(
                (scenarios, self.electric_grid_model.branches.to_flat_index()), names=["scenario", "branch"]
            )

        # Obtain individual duals.
        voltage_magnitude_vector_minimum_dual = optimization_problem.duals[
            "voltage_magnitude_vector_minimum_constraint"
        ].loc[self.electric_grid_model.timesteps, nodes] / np.concatenate(
            [np.abs(self.electric_grid_model.node_voltage_vector_reference)] * len(scenarios)
        )
        voltage_magnitude_vector_maximum_dual = (
            -1.0
            * optimization_problem.duals["voltage_magnitude_vector_maximum_constraint"].loc[
                self.electric_grid_model.timesteps, nodes
            ]
            / np.concatenate([np.abs(self.electric_grid_model.node_voltage_vector_reference)] * len(scenarios))
        )
        branch_power_magnitude_vector_1_minimum_dual = optimization_problem.duals[
            "branch_power_magnitude_vector_1_minimum_constraint"
        ].loc[self.electric_grid_model.timesteps, branches] / np.concatenate(
            [self.electric_grid_model.branch_power_vector_magnitude_reference] * len(scenarios)
        )
        branch_power_magnitude_vector_1_maximum_dual = (
            -1.0
            * optimization_problem.duals["branch_power_magnitude_vector_1_maximum_constraint"].loc[
                self.electric_grid_model.timesteps, branches
            ]
            / np.concatenate([self.electric_grid_model.branch_power_vector_magnitude_reference] * len(scenarios))
        )
        branch_power_magnitude_vector_2_minimum_dual = optimization_problem.duals[
            "branch_power_magnitude_vector_2_minimum_constraint"
        ].loc[self.electric_grid_model.timesteps, branches] / np.concatenate(
            [self.electric_grid_model.branch_power_vector_magnitude_reference] * len(scenarios)
        )
        branch_power_magnitude_vector_2_maximum_dual = (
            -1.0
            * optimization_problem.duals["branch_power_magnitude_vector_2_maximum_constraint"].loc[
                self.electric_grid_model.timesteps, branches
            ]
            / np.concatenate([self.electric_grid_model.branch_power_vector_magnitude_reference] * len(scenarios))
        )

        # Instantiate DLMP variables.
        # TODO: Consider delta connections in nodal DLMPs.
        # TODO: Consider single-phase DLMPs.
        electric_grid_energy_dlmp_node_active_power = pd.DataFrame(
            columns=nodes, index=self.electric_grid_model.timesteps, dtype=float
        )
        electric_grid_voltage_dlmp_node_active_power = pd.DataFrame(
            columns=nodes, index=self.electric_grid_model.timesteps, dtype=float
        )
        electric_grid_congestion_dlmp_node_active_power = pd.DataFrame(
            columns=nodes, index=self.electric_grid_model.timesteps, dtype=float
        )
        electric_grid_loss_dlmp_node_active_power = pd.DataFrame(
            columns=nodes, index=self.electric_grid_model.timesteps, dtype=float
        )

        electric_grid_energy_dlmp_node_reactive_power = pd.DataFrame(
            columns=nodes, index=self.electric_grid_model.timesteps, dtype=float
        )
        electric_grid_voltage_dlmp_node_reactive_power = pd.DataFrame(
            columns=nodes, index=self.electric_grid_model.timesteps, dtype=float
        )
        electric_grid_congestion_dlmp_node_reactive_power = pd.DataFrame(
            columns=nodes, index=self.electric_grid_model.timesteps, dtype=float
        )
        electric_grid_loss_dlmp_node_reactive_power = pd.DataFrame(
            columns=nodes, index=self.electric_grid_model.timesteps, dtype=float
        )

        electric_grid_energy_dlmp_der_active_power = pd.DataFrame(
            columns=ders, index=self.electric_grid_model.timesteps, dtype=float
        )
        electric_grid_voltage_dlmp_der_active_power = pd.DataFrame(
            columns=ders, index=self.electric_grid_model.timesteps, dtype=float
        )
        electric_grid_congestion_dlmp_der_active_power = pd.DataFrame(
            columns=ders, index=self.electric_grid_model.timesteps, dtype=float
        )
        electric_grid_loss_dlmp_der_active_power = pd.DataFrame(
            columns=ders, index=self.electric_grid_model.timesteps, dtype=float
        )

        electric_grid_energy_dlmp_der_reactive_power = pd.DataFrame(
            columns=ders, index=self.electric_grid_model.timesteps, dtype=float
        )
        electric_grid_voltage_dlmp_der_reactive_power = pd.DataFrame(
            columns=ders, index=self.electric_grid_model.timesteps, dtype=float
        )
        electric_grid_congestion_dlmp_der_reactive_power = pd.DataFrame(
            columns=ders, index=self.electric_grid_model.timesteps, dtype=float
        )
        electric_grid_loss_dlmp_der_reactive_power = pd.DataFrame(
            columns=ders, index=self.electric_grid_model.timesteps, dtype=float
        )

        # Obtain DLMPs.
        for timestep in self.electric_grid_model.timesteps:
            electric_grid_energy_dlmp_node_active_power.loc[timestep, :] = price_data.price_timeseries.at[
                timestep, ("active_power", "source", "source")
            ]
            electric_grid_voltage_dlmp_node_active_power.loc[timestep, :] = (
                sp.block_diag(
                    [self.linear_electric_grid_models[timestep].sensitivity_voltage_magnitude_by_power_wye_active]
                    * len(scenarios)
                ).transpose()
                @ np.transpose([voltage_magnitude_vector_minimum_dual.loc[timestep, :].values])
            ).ravel() + (
                sp.block_diag(
                    [self.linear_electric_grid_models[timestep].sensitivity_voltage_magnitude_by_power_wye_active]
                    * len(scenarios)
                ).transpose()
                @ np.transpose([voltage_magnitude_vector_maximum_dual.loc[timestep, :].values])
            ).ravel()
            electric_grid_congestion_dlmp_node_active_power.loc[timestep, :] = (
                (
                    sp.block_diag(
                        [
                            self.linear_electric_grid_models[
                                timestep
                            ].sensitivity_branch_power_1_magnitude_by_power_wye_active
                        ]
                        * len(scenarios)
                    ).transpose()
                    @ np.transpose([branch_power_magnitude_vector_1_maximum_dual.loc[timestep, :].values])
                ).ravel()
                + (
                    sp.block_diag(
                        [
                            self.linear_electric_grid_models[
                                timestep
                            ].sensitivity_branch_power_1_magnitude_by_power_wye_active
                        ]
                        * len(scenarios)
                    ).transpose()
                    @ np.transpose([branch_power_magnitude_vector_1_minimum_dual.loc[timestep, :].values])
                ).ravel()
                + (
                    sp.block_diag(
                        [
                            self.linear_electric_grid_models[
                                timestep
                            ].sensitivity_branch_power_2_magnitude_by_power_wye_active
                        ]
                        * len(scenarios)
                    ).transpose()
                    @ np.transpose([branch_power_magnitude_vector_2_maximum_dual.loc[timestep, :].values])
                ).ravel()
                + (
                    sp.block_diag(
                        [
                            self.linear_electric_grid_models[
                                timestep
                            ].sensitivity_branch_power_2_magnitude_by_power_wye_active
                        ]
                        * len(scenarios)
                    ).transpose()
                    @ np.transpose([branch_power_magnitude_vector_2_minimum_dual.loc[timestep, :].values])
                ).ravel()
            )
            electric_grid_loss_dlmp_node_active_power.loc[timestep, :] = (
                -1.0
                * np.concatenate(
                    [
                        self.linear_electric_grid_models[timestep]
                        .sensitivity_loss_active_by_power_wye_active.toarray()
                        .ravel()
                    ]
                    * len(scenarios)
                )
                * price_data.price_timeseries.at[timestep, ("active_power", "source", "source")]
                - np.concatenate(
                    [
                        self.linear_electric_grid_models[timestep]
                        .sensitivity_loss_reactive_by_power_wye_active.toarray()
                        .ravel()
                    ]
                    * len(scenarios)
                )
                * price_data.price_timeseries.at[timestep, ("reactive_power", "source", "source")]
            )

            electric_grid_energy_dlmp_node_reactive_power.loc[timestep, :] = price_data.price_timeseries.at[
                timestep, ("reactive_power", "source", "source")
            ]
            electric_grid_voltage_dlmp_node_reactive_power.loc[timestep, :] = (
                sp.block_diag(
                    [self.linear_electric_grid_models[timestep].sensitivity_voltage_magnitude_by_power_wye_reactive]
                    * len(scenarios)
                ).transpose()
                @ np.transpose([voltage_magnitude_vector_minimum_dual.loc[timestep, :].values])
            ).ravel() + (
                sp.block_diag(
                    [self.linear_electric_grid_models[timestep].sensitivity_voltage_magnitude_by_power_wye_reactive]
                    * len(scenarios)
                ).transpose()
                @ np.transpose([voltage_magnitude_vector_maximum_dual.loc[timestep, :].values])
            ).ravel()
            electric_grid_congestion_dlmp_node_reactive_power.loc[timestep, :] = (
                (
                    sp.block_diag(
                        [
                            self.linear_electric_grid_models[
                                timestep
                            ].sensitivity_branch_power_1_magnitude_by_power_wye_reactive
                        ]
                        * len(scenarios)
                    ).transpose()
                    @ np.transpose([branch_power_magnitude_vector_1_maximum_dual.loc[timestep, :].values])
                ).ravel()
                + (
                    sp.block_diag(
                        [
                            self.linear_electric_grid_models[
                                timestep
                            ].sensitivity_branch_power_1_magnitude_by_power_wye_reactive
                        ]
                        * len(scenarios)
                    ).transpose()
                    @ np.transpose([branch_power_magnitude_vector_1_minimum_dual.loc[timestep, :].values])
                ).ravel()
                + (
                    sp.block_diag(
                        [
                            self.linear_electric_grid_models[
                                timestep
                            ].sensitivity_branch_power_2_magnitude_by_power_wye_reactive
                        ]
                        * len(scenarios)
                    ).transpose()
                    @ np.transpose([branch_power_magnitude_vector_2_maximum_dual.loc[timestep, :].values])
                ).ravel()
                + (
                    sp.block_diag(
                        [
                            self.linear_electric_grid_models[
                                timestep
                            ].sensitivity_branch_power_2_magnitude_by_power_wye_reactive
                        ]
                        * len(scenarios)
                    ).transpose()
                    @ np.transpose([branch_power_magnitude_vector_2_minimum_dual.loc[timestep, :].values])
                ).ravel()
            )
            electric_grid_loss_dlmp_node_reactive_power.loc[timestep, :] = (
                -1.0
                * np.concatenate(
                    [
                        self.linear_electric_grid_models[timestep]
                        .sensitivity_loss_active_by_power_wye_reactive.toarray()
                        .ravel()
                    ]
                    * len(scenarios)
                )
                * price_data.price_timeseries.at[timestep, ("active_power", "source", "source")]
                - np.concatenate(
                    [
                        self.linear_electric_grid_models[timestep]
                        .sensitivity_loss_reactive_by_power_wye_reactive.toarray()
                        .ravel()
                    ]
                    * len(scenarios)
                )
                * price_data.price_timeseries.at[timestep, ("reactive_power", "source", "source")]
            )

            electric_grid_energy_dlmp_der_active_power.loc[timestep, :] = price_data.price_timeseries.at[
                timestep, ("active_power", "source", "source")
            ]
            electric_grid_voltage_dlmp_der_active_power.loc[timestep, :] = (
                sp.block_diag(
                    [self.linear_electric_grid_models[timestep].sensitivity_voltage_magnitude_by_der_power_active]
                    * len(scenarios)
                ).transpose()
                @ np.transpose([voltage_magnitude_vector_minimum_dual.loc[timestep, :].values])
            ).ravel() + (
                sp.block_diag(
                    [self.linear_electric_grid_models[timestep].sensitivity_voltage_magnitude_by_der_power_active]
                    * len(scenarios)
                ).transpose()
                @ np.transpose([voltage_magnitude_vector_maximum_dual.loc[timestep, :].values])
            ).ravel()
            electric_grid_congestion_dlmp_der_active_power.loc[timestep, :] = (
                (
                    sp.block_diag(
                        [
                            self.linear_electric_grid_models[
                                timestep
                            ].sensitivity_branch_power_1_magnitude_by_der_power_active
                        ]
                        * len(scenarios)
                    ).transpose()
                    @ np.transpose([branch_power_magnitude_vector_1_maximum_dual.loc[timestep, :].values])
                ).ravel()
                + (
                    sp.block_diag(
                        [
                            self.linear_electric_grid_models[
                                timestep
                            ].sensitivity_branch_power_1_magnitude_by_der_power_active
                        ]
                        * len(scenarios)
                    ).transpose()
                    @ np.transpose([branch_power_magnitude_vector_1_minimum_dual.loc[timestep, :].values])
                ).ravel()
                + (
                    sp.block_diag(
                        [
                            self.linear_electric_grid_models[
                                timestep
                            ].sensitivity_branch_power_2_magnitude_by_der_power_active
                        ]
                        * len(scenarios)
                    ).transpose()
                    @ np.transpose([branch_power_magnitude_vector_2_maximum_dual.loc[timestep, :].values])
                ).ravel()
                + (
                    sp.block_diag(
                        [
                            self.linear_electric_grid_models[
                                timestep
                            ].sensitivity_branch_power_2_magnitude_by_der_power_active
                        ]
                        * len(scenarios)
                    ).transpose()
                    @ np.transpose([branch_power_magnitude_vector_2_minimum_dual.loc[timestep, :].values])
                ).ravel()
            )
            electric_grid_loss_dlmp_der_active_power.loc[timestep, :] = (
                -1.0
                * np.concatenate(
                    [
                        self.linear_electric_grid_models[timestep]
                        .sensitivity_loss_active_by_der_power_active.toarray()
                        .ravel()
                    ]
                    * len(scenarios)
                )
                * price_data.price_timeseries.at[timestep, ("active_power", "source", "source")]
                - np.concatenate(
                    [
                        self.linear_electric_grid_models[timestep]
                        .sensitivity_loss_reactive_by_der_power_active.toarray()
                        .ravel()
                    ]
                    * len(scenarios)
                )
                * price_data.price_timeseries.at[timestep, ("reactive_power", "source", "source")]
            )

            electric_grid_energy_dlmp_der_reactive_power.loc[timestep, :] = price_data.price_timeseries.at[
                timestep, ("reactive_power", "source", "source")
            ]
            electric_grid_voltage_dlmp_der_reactive_power.loc[timestep, :] = (
                sp.block_diag(
                    [self.linear_electric_grid_models[timestep].sensitivity_voltage_magnitude_by_der_power_reactive]
                    * len(scenarios)
                ).transpose()
                @ np.transpose([voltage_magnitude_vector_minimum_dual.loc[timestep, :].values])
            ).ravel() + (
                sp.block_diag(
                    [self.linear_electric_grid_models[timestep].sensitivity_voltage_magnitude_by_der_power_reactive]
                    * len(scenarios)
                ).transpose()
                @ np.transpose([voltage_magnitude_vector_maximum_dual.loc[timestep, :].values])
            ).ravel()
            electric_grid_congestion_dlmp_der_reactive_power.loc[timestep, :] = (
                (
                    sp.block_diag(
                        [
                            self.linear_electric_grid_models[
                                timestep
                            ].sensitivity_branch_power_1_magnitude_by_der_power_reactive
                        ]
                        * len(scenarios)
                    ).transpose()
                    @ np.transpose([branch_power_magnitude_vector_1_maximum_dual.loc[timestep, :].values])
                ).ravel()
                + (
                    sp.block_diag(
                        [
                            self.linear_electric_grid_models[
                                timestep
                            ].sensitivity_branch_power_1_magnitude_by_der_power_reactive
                        ]
                        * len(scenarios)
                    ).transpose()
                    @ np.transpose([branch_power_magnitude_vector_1_minimum_dual.loc[timestep, :].values])
                ).ravel()
                + (
                    sp.block_diag(
                        [
                            self.linear_electric_grid_models[
                                timestep
                            ].sensitivity_branch_power_2_magnitude_by_der_power_reactive
                        ]
                        * len(scenarios)
                    ).transpose()
                    @ np.transpose([branch_power_magnitude_vector_2_maximum_dual.loc[timestep, :].values])
                ).ravel()
                + (
                    sp.block_diag(
                        [
                            self.linear_electric_grid_models[
                                timestep
                            ].sensitivity_branch_power_2_magnitude_by_der_power_reactive
                        ]
                        * len(scenarios)
                    ).transpose()
                    @ np.transpose([branch_power_magnitude_vector_2_minimum_dual.loc[timestep, :].values])
                ).ravel()
            )
            electric_grid_loss_dlmp_der_reactive_power.loc[timestep, :] = (
                -1.0
                * np.concatenate(
                    [
                        self.linear_electric_grid_models[timestep]
                        .sensitivity_loss_active_by_der_power_reactive.toarray()
                        .ravel()
                    ]
                    * len(scenarios)
                )
                * price_data.price_timeseries.at[timestep, ("active_power", "source", "source")]
                - np.concatenate(
                    [
                        self.linear_electric_grid_models[timestep]
                        .sensitivity_loss_reactive_by_der_power_reactive.toarray()
                        .ravel()
                    ]
                    * len(scenarios)
                )
                * price_data.price_timeseries.at[timestep, ("reactive_power", "source", "source")]
            )

        electric_grid_total_dlmp_node_active_power = (
            electric_grid_energy_dlmp_node_active_power
            + electric_grid_voltage_dlmp_node_active_power
            + electric_grid_congestion_dlmp_node_active_power
            + electric_grid_loss_dlmp_node_active_power
        )
        electric_grid_total_dlmp_node_reactive_power = (
            electric_grid_energy_dlmp_node_reactive_power
            + electric_grid_voltage_dlmp_node_reactive_power
            + electric_grid_congestion_dlmp_node_reactive_power
            + electric_grid_loss_dlmp_node_reactive_power
        )
        electric_grid_total_dlmp_der_active_power = (
            electric_grid_energy_dlmp_der_active_power
            + electric_grid_voltage_dlmp_der_active_power
            + electric_grid_congestion_dlmp_der_active_power
            + electric_grid_loss_dlmp_der_active_power
        )
        electric_grid_total_dlmp_der_reactive_power = (
            electric_grid_energy_dlmp_der_reactive_power
            + electric_grid_voltage_dlmp_der_reactive_power
            + electric_grid_congestion_dlmp_der_reactive_power
            + electric_grid_loss_dlmp_der_reactive_power
        )

        # Obtain total DLMPs in price timeseries format as in `mesmo.data_interface.PriceData.price_timeseries`.
        if len(scenarios) > 1:
            # TODO: Obtaining total DLMPs in price timeseries format is currently not possible for multiple scenarios.
            electric_grid_total_dlmp_price_timeseries = None
        else:
            electric_grid_total_dlmp_price_timeseries = pd.concat(
                [
                    price_data.price_timeseries.loc[:, ("active_power", "source", "source")].rename(
                        ("source", "source")
                    ),
                    electric_grid_total_dlmp_der_active_power,
                    price_data.price_timeseries.loc[:, ("reactive_power", "source", "source")].rename(
                        ("source", "source")
                    ),
                    electric_grid_total_dlmp_der_reactive_power,
                ],
                axis="columns",
                keys=["active_power", "active_power", "reactive_power", "reactive_power"],
                names=["commodity_type"],
            )
            # Redefine columns to avoid slicing issues.
            electric_grid_total_dlmp_price_timeseries.columns = price_data.price_timeseries.columns[
                price_data.price_timeseries.columns.isin(electric_grid_total_dlmp_price_timeseries.columns)
            ]

        return ElectricGridDLMPResults(
            electric_grid_energy_dlmp_node_active_power=electric_grid_energy_dlmp_node_active_power,
            electric_grid_voltage_dlmp_node_active_power=electric_grid_voltage_dlmp_node_active_power,
            electric_grid_congestion_dlmp_node_active_power=electric_grid_congestion_dlmp_node_active_power,
            electric_grid_loss_dlmp_node_active_power=electric_grid_loss_dlmp_node_active_power,
            electric_grid_total_dlmp_node_active_power=electric_grid_total_dlmp_node_active_power,
            electric_grid_voltage_dlmp_node_reactive_power=electric_grid_voltage_dlmp_node_reactive_power,
            electric_grid_congestion_dlmp_node_reactive_power=electric_grid_congestion_dlmp_node_reactive_power,
            electric_grid_loss_dlmp_node_reactive_power=electric_grid_loss_dlmp_node_reactive_power,
            electric_grid_energy_dlmp_node_reactive_power=electric_grid_energy_dlmp_node_reactive_power,
            electric_grid_total_dlmp_node_reactive_power=electric_grid_total_dlmp_node_reactive_power,
            electric_grid_energy_dlmp_der_active_power=electric_grid_energy_dlmp_der_active_power,
            electric_grid_voltage_dlmp_der_active_power=electric_grid_voltage_dlmp_der_active_power,
            electric_grid_congestion_dlmp_der_active_power=electric_grid_congestion_dlmp_der_active_power,
            electric_grid_loss_dlmp_der_active_power=electric_grid_loss_dlmp_der_active_power,
            electric_grid_total_dlmp_der_active_power=electric_grid_total_dlmp_der_active_power,
            electric_grid_voltage_dlmp_der_reactive_power=electric_grid_voltage_dlmp_der_reactive_power,
            electric_grid_congestion_dlmp_der_reactive_power=electric_grid_congestion_dlmp_der_reactive_power,
            electric_grid_loss_dlmp_der_reactive_power=electric_grid_loss_dlmp_der_reactive_power,
            electric_grid_energy_dlmp_der_reactive_power=electric_grid_energy_dlmp_der_reactive_power,
            electric_grid_total_dlmp_der_reactive_power=electric_grid_total_dlmp_der_reactive_power,
            electric_grid_total_dlmp_price_timeseries=electric_grid_total_dlmp_price_timeseries,
        )

    def get_optimization_results(
        self, optimization_problem: mesmo.solutions.OptimizationProblem, scenarios: typing.Union[list, pd.Index] = None
    ) -> ElectricGridOperationResults:

        # Obtain results index sets, depending on if / if not scenarios given.
        if scenarios in [None, [None]]:
            scenarios = [None]
            ders = self.electric_grid_model.ders
            nodes = self.electric_grid_model.nodes
            branches = self.electric_grid_model.branches
            loss_active = ["loss_active"]
            loss_reactive = ["loss_reactive"]
        else:
            # TODO: Check if this is correct.
            ders = (scenarios, self.electric_grid_model.ders)
            nodes = (scenarios, self.electric_grid_model.nodes)
            branches = (scenarios, self.electric_grid_model.branches)
            loss_active = scenarios
            loss_reactive = scenarios

        # Obtain results.
        der_active_power_vector_per_unit = optimization_problem.results["der_active_power_vector"].loc[
            self.electric_grid_model.timesteps, ders
        ]
        der_active_power_vector = der_active_power_vector_per_unit * np.concatenate(
            [np.real(self.electric_grid_model.der_power_vector_reference)] * len(scenarios)
        )
        der_reactive_power_vector_per_unit = optimization_problem.results["der_reactive_power_vector"].loc[
            self.electric_grid_model.timesteps, ders
        ]
        der_reactive_power_vector = der_reactive_power_vector_per_unit * np.concatenate(
            [np.imag(self.electric_grid_model.der_power_vector_reference)] * len(scenarios)
        )
        node_voltage_magnitude_vector_per_unit = optimization_problem.results["node_voltage_magnitude_vector"].loc[
            self.electric_grid_model.timesteps, nodes
        ]
        node_voltage_magnitude_vector = node_voltage_magnitude_vector_per_unit * np.concatenate(
            [np.abs(self.electric_grid_model.node_voltage_vector_reference)] * len(scenarios)
        )
        branch_power_magnitude_vector_1_per_unit = optimization_problem.results["branch_power_magnitude_vector_1"].loc[
            self.electric_grid_model.timesteps, branches
        ]
        branch_power_magnitude_vector_1 = branch_power_magnitude_vector_1_per_unit * np.concatenate(
            [self.electric_grid_model.branch_power_vector_magnitude_reference] * len(scenarios)
        )
        branch_power_magnitude_vector_2_per_unit = optimization_problem.results["branch_power_magnitude_vector_2"].loc[
            self.electric_grid_model.timesteps, branches
        ]
        branch_power_magnitude_vector_2 = branch_power_magnitude_vector_2_per_unit * np.concatenate(
            [self.electric_grid_model.branch_power_vector_magnitude_reference] * len(scenarios)
        )
        loss_active = optimization_problem.results["loss_active"].loc[self.electric_grid_model.timesteps, loss_active]
        loss_reactive = optimization_problem.results["loss_reactive"].loc[
            self.electric_grid_model.timesteps, loss_reactive
        ]

        # TODO: Obtain voltage angle and active / reactive branch power vectors.

        return ElectricGridOperationResults(
            electric_grid_model=self.electric_grid_model,
            der_active_power_vector=der_active_power_vector,
            der_active_power_vector_per_unit=der_active_power_vector_per_unit,
            der_reactive_power_vector=der_reactive_power_vector,
            der_reactive_power_vector_per_unit=der_reactive_power_vector_per_unit,
            node_voltage_magnitude_vector=node_voltage_magnitude_vector,
            node_voltage_magnitude_vector_per_unit=node_voltage_magnitude_vector_per_unit,
            branch_power_magnitude_vector_1=branch_power_magnitude_vector_1,
            branch_power_magnitude_vector_1_per_unit=branch_power_magnitude_vector_1_per_unit,
            branch_power_magnitude_vector_2=branch_power_magnitude_vector_2,
            branch_power_magnitude_vector_2_per_unit=branch_power_magnitude_vector_2_per_unit,
            loss_active=loss_active,
            loss_reactive=loss_reactive,
        )

    def get_der_power_limit_timeseries(
        self,
        der: tuple,
        der_active_power_vector: pd.DataFrame,
        der_reactive_power_vector: pd.DataFrame,
        node_voltage_magnitude_vector_minimum: np.ndarray = None,
        node_voltage_magnitude_vector_maximum: np.ndarray = None,
        branch_power_magnitude_vector_maximum: np.ndarray = None,
    ) -> pd.DataFrame:
        """Calculate power limits for given DER through maximum loadability calculation, subject to nodal voltage
        and/or branch power limits as well as the dispatch quantities of other DERs.

        Methodology (work in progress):
            1. Linear electric grid model equation:
                - 𝒔^𝑏=𝒔^(𝑏,𝑟𝑒𝑓)+𝑴^(𝑠^𝑏,𝑠^𝑑 ) 𝒔^𝑑=𝒔^(𝑏,𝑟𝑒𝑓)+𝑴^(𝑠^𝑏,𝑠^(𝑑,1) ) 𝒔^(𝑑,1)+𝑴^(𝑠^𝑏,𝑠^(𝑑,2) ) 𝒔^(𝑑,2)
                - 𝒔^𝑏 - Branch power vector
                - 𝒔^𝑑 - DER power vector; 𝒔^(𝑑,1) - DER power vector of group 1; 𝒔^(𝑑,2) - DER power vector of group 2
                - 𝑴^(𝑠^𝑏,𝑠^𝑑 ),𝑴^(𝑠^𝑏,𝑠^(𝑑,1) ),𝑴^(𝑠^𝑏,𝑠^(𝑑,2) ) 𝒔^(𝑑,2)- Sensitivity matrices
                - ()^𝑟𝑒𝑓 - Reference value / approximation point
            2. Loadability constraint:
                - 𝒔^(𝑏,𝑟𝑒𝑓)+𝑴^(𝑠^𝑏,𝑠^(𝑑,1) ) 𝒔^(𝑑,1)+𝑴^(𝑠^𝑏,𝑠^(𝑑,2) ) 𝒔^(𝑑,2)≤𝒔^(𝑏,𝑚𝑎𝑥)
                - 𝒔^(𝑏,𝑚𝑎𝑥) - Branch power limit / loading limit
            3. Reformulation for DER maximum power value:
                - 𝑴^(𝑠^𝑏,𝑠^(𝑑,2) ) 𝒔^(𝑑,2)≤𝒔^(𝑏,𝑚𝑎𝑥)−𝒔^(𝑏,𝑟𝑒𝑓)+𝑴^(𝑠^𝑏,𝑠^(𝑑,1) ) 𝒔^(𝑑,1)
                - Assume: 𝒔^(𝑑,2)∈ℝ^(1×1); Then: 𝑴^(𝑠^𝑏,𝑠^(𝑑,2) )∈ℝ^(𝑏×1)
                - 𝒔^(𝑑,2,𝑙𝑎𝑥)=diag(𝑴^(𝑠^𝑏,𝑠^(𝑑,2) ) )^(−1) (𝒔^(𝑏,𝑚𝑎𝑥)−𝒔^(𝑏,𝑟𝑒𝑓)+𝑴^(𝑠^𝑏,𝑠^(𝑑,1) ) 𝒔^(𝑑,1) )
                - With: 𝒔^(𝑑,2,𝑙𝑎𝑥)∈ℝ^(𝑏×1)
                - max((𝒔^(𝑑,2,𝑙𝑎𝑥) )^− )=𝒔^(𝑑,2,𝑚𝑖𝑛)≤𝒔^(𝑑,2)≤𝒔^(𝑑,2,𝑚𝑎𝑥)=min((𝒔^(𝑑,2,𝑙𝑎𝑥) )^+ )

        Arguments:
            ders (tuple): Index identifier of selected DER. Must be valid entry of `electric_grid_model.ders`.
            der_active_power_vector (pd.DataFrame): DER active power vector as dataframe with timesteps as index
                and DERs as columns. Must contain all other DERs aside from the selected.
            der_reactive_power_vector (pd.DataFrame): DER reactive power vector as dataframe with timesteps as index
                and DERs as columns. Must contain all other DERs aside from the selected.

        Keyword arguments:
            node_voltage_magnitude_vector_minimum (np.ndarray): Minimum nodal voltage limit vector.
            node_voltage_magnitude_vector_maximum (np.ndarray): Maximum nodal voltage limit vector.
            branch_power_magnitude_vector_maximum (np.ndarray): Maximum branch power limit vector.
        """

        # Raise error for not yet implemented functionality.
        if (node_voltage_magnitude_vector_minimum is not None) or (node_voltage_magnitude_vector_maximum is not None):
            raise NotImplementedError(
                "Maximum loadability calculation has not yet been implemented for nodal voltage limits."
            )

        # Define shorthands.
        der_index_flexible = np.array([self.electric_grid_model.ders.get_loc(der)])
        der_index_fixed = np.array(
            [index for index in range(len(self.electric_grid_model.ders)) if index not in der_index_flexible]
        )

        # Obtain branch power limit, if not set.
        if branch_power_magnitude_vector_maximum is None:
            branch_power_magnitude_vector_maximum = self.electric_grid_model.branch_power_vector_magnitude_reference

        # Calculate DER power limits.
        der_power_limit_timeseries = pd.DataFrame(np.nan, index=self.timesteps, columns=["minimum", "maximum"])
        for timestep in self.timesteps:
            linear_model = self.linear_electric_grid_models[timestep]
            der_power_laxity = sp.diags(
                linear_model.sensitivity_branch_power_1_magnitude_by_der_power_active[:, der_index_flexible]
                .toarray()
                .ravel()
                ** -1
            ) @ (
                # TODO: Revise equation to use reference power flow solution.
                np.transpose([branch_power_magnitude_vector_maximum])
                - linear_model.sensitivity_branch_power_1_magnitude_by_der_power_active[:, der_index_fixed]
                @ np.transpose([der_active_power_vector.loc[timestep, :].values[der_index_fixed]])
                - linear_model.sensitivity_branch_power_1_magnitude_by_der_power_reactive[:, der_index_fixed]
                @ np.transpose([der_reactive_power_vector.loc[timestep, :].values[der_index_fixed]])
            )
            der_power_limit_timeseries.at[timestep, "minimum"] = np.max(
                der_power_laxity[der_power_laxity < 0.0], initial=-np.inf
            )
            der_power_limit_timeseries.at[timestep, "maximum"] = np.min(
                der_power_laxity[der_power_laxity > 0.0], initial=+np.inf
            )

        return der_power_limit_timeseries
