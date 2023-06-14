import numpy as np
import pandas as pd
import scipy.sparse as sp
import mesmo
import datetime

# simulation model for calculating the energy demand
# assumptions: 1. battery charging not allowed after battery swapped at each time step
class bscs_wep_baseline_simulation_model(object):
    def __init__(
            self,
            scenario_name,
            data_set,
            swapping_demand,
            time_step,
            enable_electric_grid_model=False
    ):

        mesmo.utils.logger.info('Initializing BSCS baseline simulation model')

        # constants:
        number_of_battery_slot = data_set.bscs_data['number_of_battery_slots'].values[0]

        battery_slot_index = list(range(number_of_battery_slot))
        self.battery_slot = ['battery_slot_no_{}'.format(x) for x in battery_slot_index]

        timestep_interval_hours = (
            (time_step[1] - time_step[0]) / pd.Timedelta('1h')
        )

        # state variables initialise:

        self.rejected_ev_number = 0
        self.energy_consumption = {}
        self.battery_slot_energy = {}

        battery_slot_energy = data_set.battery_cell_data['nominal battery capacity (kWh)'].values[0] * np.ones([number_of_battery_slot])
        battery_slot_soc = battery_slot_energy / data_set.battery_cell_data['nominal battery capacity (kWh)'].values[0]

        for time_step_temp in time_step:
            if time_step_temp == time_step[-1]:
                ...
            else:

                #register for battery swapping
                register_battery_slots_for_swapping = np.array([])

                # battery swapping
                for ev_index in range(int(swapping_demand.data_number_ev_to_be_swapped_dict[time_step_temp])):
                    # check socs of all remaining batteries in battery slots
                    available_slots = np.where(battery_slot_soc >= 0.9)

                    # check if slots are empty
                    if available_slots[0].size > 0:
                        first_available_slot_index = available_slots[0][0]

                        register_battery_slots_for_swapping = np.append(register_battery_slots_for_swapping,
                                                                        [first_available_slot_index], axis=0)

                        battery_slot_energy[first_available_slot_index] = \
                            swapping_demand.data_energy_ev_to_be_swapped_dict[time_step_temp][ev_index]

                        # soc update after battery swapping
                        battery_slot_soc[first_available_slot_index] = battery_slot_energy[first_available_slot_index]/\
                            data_set.battery_cell_data['nominal battery capacity (kWh)'].values[0]
                    else:
                        self.rejected_ev_number += 1

                # battery slot charge/(discharge?)
                diff_indices = np.setdiff1d(np.arange(0, number_of_battery_slot, 1), register_battery_slots_for_swapping)

                total_charging_energy_per_time_step = 0

                for battery_slot_index in diff_indices:
                    if battery_slot_soc[battery_slot_index]<1:
                        energy_diff = abs(data_set.battery_cell_data['nominal battery capacity (kWh)'].values[0] -\
                                      battery_slot_energy[battery_slot_index])
                        if energy_diff>=timestep_interval_hours*data_set.battery_cell_data['charging power max (kW)'].values[0]:
                            battery_charge_energy = data_set.battery_cell_data['charging power max (kW)'].values[0]*timestep_interval_hours
                            battery_slot_energy[battery_slot_index] += battery_charge_energy * data_set.battery_cell_data['battery efficiency'].values[0]
                            total_charging_energy_per_time_step += battery_charge_energy
                            battery_slot_soc[battery_slot_index] = battery_slot_energy[battery_slot_index] / data_set.battery_cell_data['nominal battery capacity (kWh)'].values[0]

                        else:
                            battery_charge_energy = energy_diff
                            battery_slot_energy[battery_slot_index] += battery_charge_energy * data_set.battery_cell_data['battery efficiency'].values[0]
                            total_charging_energy_per_time_step += battery_charge_energy
                            battery_slot_soc[battery_slot_index] = battery_slot_energy[battery_slot_index] / data_set.battery_cell_data['nominal battery capacity (kWh)'].values[0]

                self.energy_consumption[time_step_temp] = total_charging_energy_per_time_step
                self.battery_slot_energy[time_step_temp] = battery_slot_energy

                #print(time_step_temp, battery_slot_energy)

        print(time_step_temp, battery_slot_soc)
        mesmo.utils.logger.info('simulation end')

# Optimisation models - energy model
class bscs_wep_optimization_model(object):
    def __init__(
            self,
            scenario_name,
            data_set,
            swapping_demand,
            time_step,
            enable_electric_grid_model=False
    ):
        soc_threshold = 0.8
        big_M_constant = data_set.battery_cell_data['nominal battery capacity (kWh)'].values[0]  # max energy capacity per battery slot
        big_M_constant_2 = soc_threshold    # max soc = 1
        big_M_constant_3 = data_set.battery_cell_data['charging power max (kW)'].values[0] # max charge/discharge power

        mesmo.utils.logger.info('Initializing BSCS wholesale market optimisation model...')

        # Obtain DER & grid model objects.
        self.der_model_set = mesmo.der_models.DERModelSet(scenario_name)

        # grid model - not used
        # linear_electric_grid_model_set = mesmo.electric_grid_models.LinearElectricGridModelSet(scenario_name)

        # settings
        # reserved for stochastic optimisation
        self.scenarios = ['deterministic']

        # battery slot index
        number_of_battery_slot = data_set.bscs_data['number_of_battery_slots'].values[0]
        battery_slot_index = list(range(number_of_battery_slot))
        self.battery_slot = ['battery_slot_no_{}'.format(x) for x in battery_slot_index]

        # time steps for electricity market participation
        self.timesteps = time_step

        # for SOC constraints
        self.timesteps_minus = time_step[0:(time_step.size-1)]
        self.timesteps_plus = time_step[1:time_step.size]

        # Obtain timestep interval in hours, for conversion of power to energy.
        timestep_interval_hours = (
            (self.timesteps[1] - self.timesteps[0]) / pd.Timedelta('1h')
        )

        # Instantiate optimization problem.
        self.optimization_problem = mesmo.utils.OptimizationProblem()

        self.optimization_problem.define_variable(
            "battery_charge_power",
            scenario=self.scenarios,
            timestep=self.timesteps_minus,
            battery_slot_indices=self.battery_slot,
        )

        self.optimization_problem.define_variable(
            "battery_discharge_power",
            scenario=self.scenarios,
            timestep=self.timesteps_minus,
            battery_slot_indices=self.battery_slot,
        )

        self.optimization_problem.define_variable(
            "battery_slot_energy",
            scenario=self.scenarios,
            timestep=self.timesteps,
            battery_slot_indices=self.battery_slot,
        )

        self.optimization_problem.define_variable(
            "battery_slot_soc",
            scenario=self.scenarios,
            timestep=self.timesteps,
            battery_slot_indices=self.battery_slot,
        )

        # Define price arbitrage variables.
        self.optimization_problem.define_variable(
            'swapping_station_total_energy_demand',
            timestep=self.timesteps_minus,
            scenario=self.scenarios,
        )

        # Define A_t matrix
        # number of incoming batteries:  swapping_demand.data_number_ev_to_be_swapped_dict[time_step[3]]
        for time_step_temp in self.timesteps_minus:
            self.optimization_problem.define_variable(
                'A_matrix',
                variable_type='binary',
                timestep=time_step_temp,
                scenario=self.scenarios,
                battery_slot_indices=self.battery_slot,
                ev_indices=list(range(int(swapping_demand.data_number_ev_to_be_swapped_dict[time_step_temp]))),
            )

            self.optimization_problem.define_variable(
                'swapping_points',
                timestep=time_step_temp,
                scenario=self.scenarios,
            )

            self.optimization_problem.define_variable(
                'swapping_events_per_slot',
                timestep=time_step_temp,
                battery_slot_indices=self.battery_slot,
                scenario=self.scenarios,
            )

        # Define Z_t matrix
        #     self.optimization_problem.define_variable(
        #         'Z_matrix',
        #         timestep=time_step_temp,
        #         scenario=self.scenarios,
        #         battery_slot_indices=self.battery_slot,
        #         ev_indices=list(range(int(swapping_demand.data_number_ev_to_be_swapped_dict[time_step_temp]))),
        #     )

            self.optimization_problem.define_variable(
                'Z_matrix',
                timestep=time_step_temp,
                scenario=self.scenarios,
                battery_slot_indices=self.battery_slot,
            )

        # Define constraints
        # Define energy constraints
        mesmo.utils.logger.info('Define energy constraints 1a)')

        for (timestep_temp_minus, timestep_temp_plus) in zip(self.timesteps_minus, self.timesteps_plus):
            # number of incoming EVs per timestep
            number_ev_per_step = swapping_demand.data_SOC_ev_to_be_swapped_dict[timestep_temp_minus].size
            A_coefficient_matrix = np.zeros([number_of_battery_slot, number_of_battery_slot*number_ev_per_step])
            Z_coefficient_matrix = np.zeros([number_of_battery_slot, number_of_battery_slot*number_ev_per_step])

            for index_temp in range(number_of_battery_slot):

                A_coefficient_matrix[index_temp, index_temp*number_ev_per_step: (index_temp+1)*number_ev_per_step] = \
                    swapping_demand.data_energy_ev_to_be_swapped_dict[timestep_temp_minus]

                Z_coefficient_matrix[index_temp, index_temp*number_ev_per_step: (index_temp+1)*number_ev_per_step] = -1

            A_coefficient_matrix = sp.csr_matrix(A_coefficient_matrix)
            Z_coefficient_matrix = sp.csr_matrix(Z_coefficient_matrix)

            ## Debug print('debug')
            # indices = self.optimization_problem.get_variable_index(name='Z_matrix', timestep=timestep_temp_minus)
            # var_res = self.optimization_problem.variables.iloc[indices]

            # constraint 1)
            self.optimization_problem.define_constraint(
                (
                    "variable",
                    1,
                    dict(
                            name="battery_slot_energy",
                            scenario=self.scenarios,
                            timestep=timestep_temp_plus,
                    )
                ),
                "==",
                (
                    "variable",
                    A_coefficient_matrix,
                    dict(
                            name="A_matrix",
                            scenario=self.scenarios,
                            timestep=timestep_temp_minus,
                    )
                ),
                (
                    "variable",
                    1,
                    dict(
                            name="battery_slot_energy",
                            scenario=self.scenarios,
                            timestep=timestep_temp_minus,
                    )
                ),
                (
                    "variable",
                    -1, #Z_coefficient_matrix,
                    dict(
                        name="Z_matrix",
                        scenario=self.scenarios,
                        timestep=timestep_temp_minus,
                    )
                ),
                (
                    "variable",
                    data_set.battery_cell_data['battery efficiency'].values[0] * timestep_interval_hours,
                    dict(
                            name="battery_charge_power", scenario=self.scenarios, timestep=timestep_temp_minus,
                            battery_slot_indices=self.battery_slot,
                    ),
                ),
                (
                    "variable",
                    -1/data_set.battery_cell_data['battery efficiency'].values[0] * timestep_interval_hours,
                    dict(
                            name="battery_discharge_power", scenario=self.scenarios, timestep=timestep_temp_minus,
                            battery_slot_indices=self.battery_slot
                    ),
                ),
                broadcast=["scenario"],
            )


        mesmo.utils.logger.info('Define constraints 1b) - 1e)')
        # constraint 9a)
        self.optimization_problem.define_constraint(
            (
                "variable",
                1,
                dict(
                        name="battery_slot_energy",
                        scenario=self.scenarios,
                        timestep=self.timesteps,
                        battery_slot_indices=self.battery_slot,
                )
            ),
            "<=",
            (
                "constant",
                data_set.battery_cell_data["max battery capacity (kWh)"].values[0],
                dict(
                        scenario=self.scenarios,
                        timestep=self.timesteps,
                        battery_slot_indices=self.battery_slot,
                )
            ),
            broadcast=["scenario", "timestep", "battery_slot_indices"],
        )

        # constraint 9b) part 1
        for time_step_temp in self.timesteps_minus:
            self.optimization_problem.define_constraint(
                (
                    "variable",
                    1,
                    dict(
                        name="Z_matrix",
                        timestep=time_step_temp,
                        scenario=self.scenarios,
                        battery_slot_indices=self.battery_slot,
                    )
                ),
                ">=",
                (
                    "constant",
                    0,
                ),
                broadcast=["scenario", "battery_slot_indices"],
            )

        # constraint 9b) part 2
        for time_step_temp in self.timesteps_minus:
            self.optimization_problem.define_constraint(
                (
                    "variable",
                    1,
                    dict(
                        name="Z_matrix",
                        timestep=time_step_temp,
                        scenario=self.scenarios,
                        battery_slot_indices=self.battery_slot,
                    )
                ),
                "<=",
                (
                    "variable",
                    1,
                    dict(
                        name="battery_slot_energy",
                        timestep=time_step_temp,
                        scenario=self.scenarios,
                        battery_slot_indices=self.battery_slot,
                    )
                ),
                broadcast=["scenario", "battery_slot_indices"],
            )

        # Define constraint 9c)
        for time_step_temp in self.timesteps_minus:
            self.optimization_problem.define_constraint(
                (
                    "variable",
                    1,
                    dict(
                        name="Z_matrix",
                        timestep=time_step_temp,
                        scenario=self.scenarios,
                        battery_slot_indices=self.battery_slot
                    )
                ),
                "<=",
                (
                    "variable",
                    big_M_constant*np.ones([1, int(swapping_demand.data_number_ev_to_be_swapped_dict[time_step_temp])]),
                    dict(
                        name="A_matrix",
                        timestep=time_step_temp,
                        scenario=self.scenarios,
                        battery_slot_indices=self.battery_slot,
                    )
                ),
                broadcast=["scenario", "battery_slot_indices"],
            )

        # Define 9d)
        for time_step_temp in self.timesteps_minus:
            self.optimization_problem.define_constraint(
                (
                    "variable",
                    1,
                    dict(
                            name="Z_matrix",
                            timestep=time_step_temp,
                            scenario=self.scenarios,
                            battery_slot_indices=self.battery_slot,
                    )
                ),
                ">=",
                (
                    "variable",
                    1,
                    dict(
                            name="battery_slot_energy",
                            timestep=time_step_temp,
                            scenario=self.scenarios,
                            battery_slot_indices=self.battery_slot,
                    )
                ),
                (
                    "constant",
                    -big_M_constant,
                    dict(
                            scenario=self.scenarios,
                            timestep=time_step_temp,
                            battery_slot_indices=self.battery_slot,
                    )
                ),
                (
                    "variable",
                    big_M_constant*np.ones([1, int(swapping_demand.data_number_ev_to_be_swapped_dict[time_step_temp])]),
                    dict(
                            name="A_matrix",
                            timestep=time_step_temp,
                            scenario=self.scenarios,
                            battery_slot_indices=self.battery_slot,
                    )
                ),
                broadcast=["scenario", "battery_slot_indices"],
            )

        mesmo.utils.logger.info('Define constraints 2a) - 2e)')
        # constraint 10)
        for time_step_temp in self.timesteps_minus:
            self.optimization_problem.define_constraint(
                (
                    "variable",
                    np.ones([1, number_of_battery_slot]),
                    dict(
                        name="A_matrix",
                        timestep=time_step_temp,
                        scenario=self.scenarios,
                        battery_slot_indices=self.battery_slot,
                        ev_indices=list(range(int(swapping_demand.data_number_ev_to_be_swapped_dict[time_step_temp]))),
                    )
                ),
                "<=",
                (
                    "constant",
                    1,
                    dict(
                        timestep=time_step_temp,
                        scenario=self.scenarios,
                        battery_slot_indices=self.battery_slot,
                        ev_indices=list(range(int(swapping_demand.data_number_ev_to_be_swapped_dict[time_step_temp]))),
                    )
                ),
                broadcast=["scenario", "ev_indices"],
            )

        for time_step_temp in self.timesteps_minus:
            self.optimization_problem.define_constraint(
                (
                    "variable",
                    np.ones([1, int(swapping_demand.data_number_ev_to_be_swapped_dict[time_step_temp])]),
                    dict(
                            name="A_matrix",
                            timestep=time_step_temp,
                            scenario=self.scenarios,
                            battery_slot_indices=self.battery_slot,
                    )
                ),
                ">=",
                (
                    "constant",
                    0,
                    dict(
                            timestep=time_step_temp,
                            scenario=self.scenarios,
                            battery_slot_indices=self.battery_slot,
                    )
                ),
                broadcast=["scenario", "battery_slot_indices"],
            )

            # Debug
            # indices = self.optimization_problem.get_variable_index(name='Z_matrix', timestep=timestep_temp_minus)
            # var_res = self.optimization_problem.variables.iloc[indices]

        for time_step_temp in self.timesteps_minus:
            #   constraint 14)
            self.optimization_problem.define_constraint(
                (
                    "variable",
                    1,
                    dict(
                            name="battery_slot_soc",
                            timestep=time_step_temp,
                            scenario=self.scenarios,
                            battery_slot_indices=self.battery_slot,
                    )
                ),
                ">=",
                (
                    "variable",
                    big_M_constant_2*np.ones([1, int(swapping_demand.data_number_ev_to_be_swapped_dict[time_step_temp])]),
                    dict(
                            name="A_matrix",
                            timestep=time_step_temp,
                            scenario=self.scenarios,
                            battery_slot_indices=self.battery_slot,
                    )
                ),
                broadcast=["scenario", "battery_slot_indices"],
            )

        # constraint 19)
        self.optimization_problem.define_constraint(
            (
                "variable",
                1,
                dict(
                    name="battery_slot_soc",
                    timestep=self.timesteps,
                    scenario=self.scenarios,
                    battery_slot_indices=self.battery_slot,
                )
            ),
            "==",
            (
                "variable",
                1/data_set.battery_cell_data['nominal battery capacity (kWh)'].values[0],
                dict(
                    name="battery_slot_energy",
                    timestep=self.timesteps,
                    scenario=self.scenarios,
                    battery_slot_indices=self.battery_slot,
                )
            ),
            broadcast=["scenario", "timestep", "battery_slot_indices"],
        )

        # constraint 18b) set initial energy at time step 0
        self.optimization_problem.define_constraint(
            (
                "variable",
                1,
                dict(
                    name="battery_slot_energy",
                    timestep=self.timesteps[0],
                    scenario=self.scenarios,
                    battery_slot_indices=self.battery_slot,
                )
            ),
            "==",
            (
                "constant",
                data_set.battery_cell_data['nominal battery capacity (kWh)'].values[0],
            ),
            broadcast=["scenario", "battery_slot_indices"],
        )

        # constraint 20a)
        self.optimization_problem.define_constraint(
            (
                "variable",
                1,
                dict(
                    name="battery_charge_power",
                    timestep=self.timesteps_minus,
                    scenario=self.scenarios,
                    battery_slot_indices=self.battery_slot,
                )
            ),
            ">=",
            (
                "constant",
                0,
            ),
            broadcast=["scenario", "timestep", "battery_slot_indices"],
        )

        self.optimization_problem.define_constraint(
            (
                "variable",
                1,
                dict(
                    name="battery_charge_power",
                    timestep=self.timesteps_minus,
                    scenario=self.scenarios,
                    battery_slot_indices=self.battery_slot,
                )
            ),
            "<=",
            (
                "constant",
                data_set.battery_cell_data['charging power max (kW)'].values[0],
            ),
            broadcast=["scenario", "timestep", "battery_slot_indices"],
        )

        self.optimization_problem.define_constraint(
            (
                "variable",
                1,
                dict(
                    name="battery_discharge_power",
                    timestep=self.timesteps_minus,
                    scenario=self.scenarios,
                    battery_slot_indices=self.battery_slot,
                )
            ),
            ">=",
            (
                "constant",
                0,
            ),
            broadcast=["scenario", "timestep", "battery_slot_indices"],
        )

        self.optimization_problem.define_constraint(
            (
                "variable",
                1,
                dict(
                    name="battery_discharge_power",
                    timestep=self.timesteps_minus,
                    scenario=self.scenarios,
                    battery_slot_indices=self.battery_slot,
                )
            ),
            "<=",
            (
                "constant",
                data_set.battery_cell_data['charging power max (kW)'].values[0],
            ),
            broadcast=["scenario", "timestep", "battery_slot_indices"],
        )

        mesmo.utils.logger.info('Define constraints 20b)')
        # constraint 8 total energy per swapping/charging station
        self.optimization_problem.define_constraint(
            (
                "variable",
                1,
                dict(
                    name="swapping_station_total_energy_demand",
                    scenario=self.scenarios,
                    timestep=self.timesteps_minus,
                )
            ),
            "==",
            (
                "variable",
                -1*np.ones([1, number_of_battery_slot]),
                dict(
                        name="battery_discharge_power",
                        scenario=self.scenarios,
                        timestep=self.timesteps_minus,
                )
            ),
            (
                "variable",
                np.ones([1, number_of_battery_slot]),
                dict(
                    name="battery_charge_power",
                    scenario=self.scenarios,
                    timestep=self.timesteps_minus,
                )
            ),
            broadcast=["scenario", "timestep"],
        )

        # constraint final soc
        self.optimization_problem.define_constraint(
            (
                "variable",
                1,
                dict(
                    name="battery_slot_soc",
                    timestep=self.timesteps[-1],
                    scenario=self.scenarios,
                    battery_slot_indices=self.battery_slot,
                )
            ),
            ">=",
            (
                "constant",
                0.1,
            ),
            broadcast=["scenario", "battery_slot_indices"],
        )

        i = 0
        #  constraint 10
        for time_step_temp in self.timesteps_minus:
            self.optimization_problem.define_constraint(
                (
                    "variable",
                    np.ones([1, int(swapping_demand.data_number_ev_to_be_swapped_dict[time_step_temp])]),
                    dict(
                            name="A_matrix",
                            timestep=time_step_temp,
                            scenario=self.scenarios,
                            battery_slot_indices=self.battery_slot,
                            ev_indices=list(
                            range(int(swapping_demand.data_number_ev_to_be_swapped_dict[time_step_temp]))),
                    )
                ),
                "==",
                (
                    "variable",
                    1,
                    dict(
                            name="swapping_events_per_slot",
                            timestep=time_step_temp,
                            scenario=self.scenarios,
                            battery_slot_indices=self.battery_slot,
                    )
                ),
                broadcast=["scenario", "battery_slot_indices"],
            )

            score_parameter = number_of_battery_slot*swapping_demand.data_number_ev_to_be_swapped_dict[time_step_temp]

            # magic function to set the start parameter in linspace(), it seems to impact the solution time a lot
            # (score_parameter+1)/13 seems to work well if rand(1,3) for incoming evs
            # (score_parameter + swapping_demand.data_number_ev_to_be_swapped_dict[
            #    time_step_temp]) / number_of_battery_slot seems to work well if rand(1,3) for incoming evs

            i += 0
            # swapping scores calculation
            self.optimization_problem.define_constraint(
                (
                    "variable",
                    np.linspace(i+((score_parameter+swapping_demand.data_number_ev_to_be_swapped_dict[time_step_temp])/number_of_battery_slot),
                                i+1, int(score_parameter), endpoint=True),
                    dict(
                            name="A_matrix",
                            timestep=time_step_temp,
                            scenario=self.scenarios,
                            battery_slot_indices=self.battery_slot,
                            ev_indices=list(
                            range(int(swapping_demand.data_number_ev_to_be_swapped_dict[time_step_temp]))),
                    )
                ),
                "==",
                (
                    "variable",
                    1,
                    dict(
                            name="swapping_points",
                            timestep=time_step_temp,
                            scenario=self.scenarios,
                    )
                ),
            )

        # constraint 11
        # a)
        for time_step_temp in self.timesteps_minus:
            self.optimization_problem.define_constraint(
                (
                    "variable",
                    1,
                    dict(
                        name="battery_charge_power",
                        timestep=time_step_temp,
                        scenario=self.scenarios,
                        battery_slot_indices=self.battery_slot,
                    )
                ),
                "<=",
                (
                    "constant",
                    big_M_constant_3,
                ),
                (
                    "variable",
                    -big_M_constant_3*np.ones([1, int(swapping_demand.data_number_ev_to_be_swapped_dict[time_step_temp])]),
                    dict(
                            name="A_matrix",
                            timestep=time_step_temp,
                            scenario=self.scenarios,
                            battery_slot_indices=self.battery_slot,
                    )
                ),
                broadcast=["scenario", "battery_slot_indices"],
            )

            # b)
            self.optimization_problem.define_constraint(
                (
                    "variable",
                    1,
                    dict(
                        name="battery_discharge_power",
                        timestep=time_step_temp,
                        scenario=self.scenarios,
                        battery_slot_indices=self.battery_slot,
                    )
                ),
                "<=",
                (
                    "constant",
                    big_M_constant_3,
                ),
                (
                    "variable",
                    -big_M_constant_3*np.ones([1, int(swapping_demand.data_number_ev_to_be_swapped_dict[time_step_temp])]),
                    dict(
                            name="A_matrix",
                            timestep=time_step_temp,
                            scenario=self.scenarios,
                            battery_slot_indices=self.battery_slot,
                    )
                ),
                broadcast=["scenario", "battery_slot_indices"],
            )


        # Note: 1. trivial case if set as <=1000
        # 2. Too large value, e.g., >=8000 will cause long computation time
        # 3。 Medium-scale values leads to partial swapping of total demand. E.g., 4000 - acceptance rate of 30%
        swapping_score = 7000

        mesmo.utils.logger.info('Define objective function')
        # wep data selection
        wep = data_set.wholesale_electricity_price_data[0:self.timesteps_minus.size]["WEP ($/MWh)"]
        wep = wep.values

        self.wep = wep
        self.timestep_interval_hours = timestep_interval_hours

        self.optimization_problem.define_objective(
            (
                'variable',
                (wep * timestep_interval_hours),
                dict(name='swapping_station_total_energy_demand', timestep=self.timesteps_minus)
            ),
            (
                'variable',
                -swapping_score*np.ones([1, self.timesteps_minus.size]),
                dict(name='swapping_points', timestep=self.timesteps_minus)
            ),
            # (
            #     'variable',
            #     (data_set.battery_cell_data['battery degradation coefficient ($/kWh^2)'].values[0] * sp.block_diag(
            #         [sp.diags(np.random.rand(number_of_battery_slot))] * len(self.timesteps_minus)
            #      )),
            #     dict(name='battery_charge_power'), dict(name='battery_charge_power')
            # ),
            # (
            #     'variable',
            #     (
            #         data_set.battery_cell_data['battery degradation coefficient ($/kWh^2)'].values[0] * sp.block_diag(
            #         [sp.diags(np.random.rand(number_of_battery_slot))] * len(self.timesteps_minus)
            #     )
            #     ),
            #     dict(name='battery_discharge_power'), dict(name='battery_discharge_power')
            # ),
        )

        mesmo.utils.logger.info('BSCS WEP model defined!')

# Optimisation models - energy + contingency reserve model
class bscs_contigency_reserve_optimization_model(object):
    def __init__(
            self,
            scenario_name,
            data_set,
            swapping_demand,
            time_step,
            enable_electric_grid_model=False
    ):
        soc_threshold = 0.8
        constant_minimal_reserve = 100  # 100 kWh
        number_of_battery_slot = data_set.bscs_data['number_of_battery_slots'].values[0]

        big_M_constant = data_set.battery_cell_data['nominal battery capacity (kWh)'].values[0]  # max energy capacity per battery slot
        big_M_constant_2 = soc_threshold    # max soc = 1
        big_M_constant_3 = data_set.battery_cell_data['charging power max (kW)'].values[0] # max charge/discharge power
        big_M_constant_4 = data_set.battery_cell_data['charging power max (kW)'].values[0]*number_of_battery_slot

        mesmo.utils.logger.info('Initializing BSCS wholesale market optimisation model...')

        # Obtain DER & grid model objects.
        self.der_model_set = mesmo.der_models.DERModelSet(scenario_name)

        # settings
        # reserved for stochastic optimisation
        self.scenarios = ['maximal_reserve_activation', 'no_reserve_activation']

        # battery slot index
        battery_slot_index = list(range(number_of_battery_slot))
        self.battery_slot = ['battery_slot_no_{}'.format(x) for x in battery_slot_index]

        # time steps for electricity market participation
        self.timesteps = time_step

        # for SOC constraints
        self.timesteps_minus = time_step[0:(time_step.size-1)]
        self.timesteps_plus = time_step[1:time_step.size]

        # Obtain timestep interval in hours, for conversion of power to energy.
        timestep_interval_hours = (
            (self.timesteps[1] - self.timesteps[0]) / pd.Timedelta('1h')
        )

        # Instantiate optimization problem.
        self.optimization_problem = mesmo.utils.OptimizationProblem()

        self.optimization_problem.define_variable(
            "battery_charge_power",
            scenario=self.scenarios,
            timestep=self.timesteps_minus,
            battery_slot_indices=self.battery_slot,
        )

        self.optimization_problem.define_variable(
            "battery_discharge_power",
            scenario=self.scenarios,
            timestep=self.timesteps_minus,
            battery_slot_indices=self.battery_slot,
        )

        self.optimization_problem.define_variable(
            "battery_slot_energy",
            scenario=self.scenarios,
            timestep=self.timesteps,
            battery_slot_indices=self.battery_slot,
        )

        self.optimization_problem.define_variable(
            "battery_slot_soc",
            scenario=self.scenarios,
            timestep=self.timesteps,
            battery_slot_indices=self.battery_slot,
        )

        # Define price arbitrage variables.
        self.optimization_problem.define_variable(
            'swapping_station_total_energy_demand',
            timestep=self.timesteps_minus,
            scenario=self.scenarios[0],
        )

        self.optimization_problem.define_variable(
            'contingency_reserve',
            timestep=self.timesteps_minus,
            scenario=self.scenarios[0],
        )

        self.optimization_problem.define_variable(
            'contingency_reserve_binary',
            variable_type='binary',
            timestep=self.timesteps_minus,
            scenario=self.scenarios[0],
        )

        # Define A_t matrix
        # number of incoming batteries:  swapping_demand.data_number_ev_to_be_swapped_dict[time_step[3]]
        for time_step_temp in self.timesteps_minus:
            self.optimization_problem.define_variable(
                'A_matrix',
                variable_type='binary',
                timestep=time_step_temp,
                scenario=self.scenarios,
                battery_slot_indices=self.battery_slot,
                ev_indices=list(range(int(swapping_demand.data_number_ev_to_be_swapped_dict[time_step_temp]))),
            )

            self.optimization_problem.define_variable(
                'swapping_points',
                timestep=time_step_temp,
                scenario=self.scenarios,
            )

            self.optimization_problem.define_variable(
                'swapping_events_per_slot',
                timestep=time_step_temp,
                battery_slot_indices=self.battery_slot,
                scenario=self.scenarios,
            )

            self.optimization_problem.define_variable(
                'Z_matrix',
                timestep=time_step_temp,
                scenario=self.scenarios,
                battery_slot_indices=self.battery_slot,
            )

        # Define constraints
        # Define energy constraints
        mesmo.utils.logger.info('Define energy constraints 1a)')

        for (timestep_temp_minus, timestep_temp_plus) in zip(self.timesteps_minus, self.timesteps_plus):
            # number of incoming EVs per timestep
            number_ev_per_step = swapping_demand.data_SOC_ev_to_be_swapped_dict[timestep_temp_minus].size
            A_coefficient_matrix = np.zeros([number_of_battery_slot, number_of_battery_slot*number_ev_per_step])

            for index_temp in range(number_of_battery_slot):

                A_coefficient_matrix[index_temp, index_temp*number_ev_per_step: (index_temp+1)*number_ev_per_step] = \
                    swapping_demand.data_energy_ev_to_be_swapped_dict[timestep_temp_minus]

            A_coefficient_matrix = sp.csr_matrix(A_coefficient_matrix)

            # constraint 1)
            self.optimization_problem.define_constraint(
                (
                    "variable",
                    1,
                    dict(
                            name="battery_slot_energy",
                            scenario=self.scenarios,
                            timestep=timestep_temp_plus,
                    )
                ),
                "==",
                (
                    "variable",
                    A_coefficient_matrix,
                    dict(
                            name="A_matrix",
                            scenario=self.scenarios,
                            timestep=timestep_temp_minus,
                    )
                ),
                (
                    "variable",
                    1,
                    dict(
                            name="battery_slot_energy",
                            scenario=self.scenarios,
                            timestep=timestep_temp_minus,
                    )
                ),
                (
                    "variable",
                    -1,
                    dict(
                        name="Z_matrix",
                        scenario=self.scenarios,
                        timestep=timestep_temp_minus,
                    )
                ),
                (
                    "variable",
                    data_set.battery_cell_data['battery efficiency'].values[0] * timestep_interval_hours,
                    dict(
                            name="battery_charge_power", scenario=self.scenarios, timestep=timestep_temp_minus,
                            battery_slot_indices=self.battery_slot,
                    ),
                ),
                (
                    "variable",
                    -1/data_set.battery_cell_data['battery efficiency'].values[0] * timestep_interval_hours,
                    dict(
                            name="battery_discharge_power", scenario=self.scenarios, timestep=timestep_temp_minus,
                            battery_slot_indices=self.battery_slot
                    ),
                ),
                broadcast=["scenario"],
            )


        mesmo.utils.logger.info('Define constraints 9a) - 9d)')
        # constraint 9a)
        self.optimization_problem.define_constraint(
            (
                "variable",
                1,
                dict(
                        name="battery_slot_energy",
                        scenario=self.scenarios,
                        timestep=self.timesteps,
                        battery_slot_indices=self.battery_slot,
                )
            ),
            "<=",
            (
                "constant",
                data_set.battery_cell_data["max battery capacity (kWh)"].values[0],
                dict(
                        scenario=self.scenarios,
                        timestep=self.timesteps,
                        battery_slot_indices=self.battery_slot,
                )
            ),
            broadcast=["scenario", "timestep", "battery_slot_indices"],
        )

        # constraint 9b) part 1
        for time_step_temp in self.timesteps_minus:
            self.optimization_problem.define_constraint(
                (
                    "variable",
                    1,
                    dict(
                        name="Z_matrix",
                        timestep=time_step_temp,
                        scenario=self.scenarios,
                        battery_slot_indices=self.battery_slot,
                    )
                ),
                ">=",
                (
                    "constant",
                    0,
                ),
                broadcast=["scenario", "battery_slot_indices"],
            )

        # constraint 9b) part 2
        for time_step_temp in self.timesteps_minus:
            self.optimization_problem.define_constraint(
                (
                    "variable",
                    1,
                    dict(
                        name="Z_matrix",
                        timestep=time_step_temp,
                        scenario=self.scenarios,
                        battery_slot_indices=self.battery_slot,
                    )
                ),
                "<=",
                (
                    "variable",
                    1,
                    dict(
                        name="battery_slot_energy",
                        timestep=time_step_temp,
                        scenario=self.scenarios,
                        battery_slot_indices=self.battery_slot,
                    )
                ),
                broadcast=["scenario", "battery_slot_indices"],
            )

        # Define constraint 9c)
        for time_step_temp in self.timesteps_minus:
            self.optimization_problem.define_constraint(
                (
                    "variable",
                    1,
                    dict(
                        name="Z_matrix",
                        timestep=time_step_temp,
                        scenario=self.scenarios,
                        battery_slot_indices=self.battery_slot
                    )
                ),
                "<=",
                (
                    "variable",
                    big_M_constant*np.ones([1, int(swapping_demand.data_number_ev_to_be_swapped_dict[time_step_temp])]),
                    dict(
                        name="A_matrix",
                        timestep=time_step_temp,
                        scenario=self.scenarios,
                        battery_slot_indices=self.battery_slot,
                    )
                ),
                broadcast=["scenario", "battery_slot_indices"],
            )

        # Define 9d)
        for time_step_temp in self.timesteps_minus:
            self.optimization_problem.define_constraint(
                (
                    "variable",
                    1,
                    dict(
                            name="Z_matrix",
                            timestep=time_step_temp,
                            scenario=self.scenarios,
                            battery_slot_indices=self.battery_slot,
                    )
                ),
                ">=",
                (
                    "variable",
                    1,
                    dict(
                            name="battery_slot_energy",
                            timestep=time_step_temp,
                            scenario=self.scenarios,
                            battery_slot_indices=self.battery_slot,
                    )
                ),
                (
                    "constant",
                    -big_M_constant,
                    dict(
                            scenario=self.scenarios,
                            timestep=time_step_temp,
                            battery_slot_indices=self.battery_slot,
                    )
                ),
                (
                    "variable",
                    big_M_constant*np.ones([1, int(swapping_demand.data_number_ev_to_be_swapped_dict[time_step_temp])]),
                    dict(
                            name="A_matrix",
                            timestep=time_step_temp,
                            scenario=self.scenarios,
                            battery_slot_indices=self.battery_slot,
                    )
                ),
                broadcast=["scenario", "battery_slot_indices"],
            )

        # constraint 10)
        for time_step_temp in self.timesteps_minus:
            self.optimization_problem.define_constraint(
                (
                    "variable",
                    np.ones([1, number_of_battery_slot]),
                    dict(
                            name="A_matrix",
                            timestep=time_step_temp,
                            scenario=self.scenarios,
                            battery_slot_indices=self.battery_slot,
                            ev_indices=list(range(int(swapping_demand.data_number_ev_to_be_swapped_dict[time_step_temp]))),
                    )
                ),
                "<=",
                (
                    "constant",
                    1,
                    dict(
                            timestep=time_step_temp,
                            scenario=self.scenarios,
                            battery_slot_indices=self.battery_slot,
                            ev_indices=list(range(int(swapping_demand.data_number_ev_to_be_swapped_dict[time_step_temp]))),
                    )
                ),
                broadcast=["scenario", "ev_indices"],
            )

       # constraint 17
       # a)
        for time_step_temp in self.timesteps_minus:
            self.optimization_problem.define_constraint(
                (
                    "variable",
                    1,
                    dict(
                        name="battery_charge_power",
                        timestep=time_step_temp,
                        scenario=self.scenarios,
                        battery_slot_indices=self.battery_slot,
                    )
                ),
                "<=",
                (
                    "constant",
                    big_M_constant_3,
                ),
                (
                    "variable",
                    -big_M_constant_3 * np.ones(
                        [1, int(swapping_demand.data_number_ev_to_be_swapped_dict[time_step_temp])]),
                    dict(
                        name="A_matrix",
                        timestep=time_step_temp,
                        scenario=self.scenarios,
                        battery_slot_indices=self.battery_slot,
                    )
                ),
                broadcast=["scenario", "battery_slot_indices"],
            )

            # b)
            self.optimization_problem.define_constraint(
                (
                    "variable",
                    1,
                    dict(
                        name="battery_discharge_power",
                        timestep=time_step_temp,
                        scenario=self.scenarios,
                        battery_slot_indices=self.battery_slot,
                    )
                ),
                "<=",
                (
                    "constant",
                    big_M_constant_3,
                ),
                (
                    "variable",
                    -big_M_constant_3 * np.ones(
                        [1, int(swapping_demand.data_number_ev_to_be_swapped_dict[time_step_temp])]),
                    dict(
                        name="A_matrix",
                        timestep=time_step_temp,
                        scenario=self.scenarios,
                        battery_slot_indices=self.battery_slot,
                    )
                ),
                broadcast=["scenario", "battery_slot_indices"],
            )

        for time_step_temp in self.timesteps_minus:
            self.optimization_problem.define_constraint(
                (
                    "variable",
                    np.ones([1, int(swapping_demand.data_number_ev_to_be_swapped_dict[time_step_temp])]),
                    dict(
                            name="A_matrix",
                            timestep=time_step_temp,
                            scenario=self.scenarios,
                            battery_slot_indices=self.battery_slot,
                    )
                ),
                ">=",
                (
                    "constant",
                    0,
                    dict(
                            timestep=time_step_temp,
                            scenario=self.scenarios,
                            battery_slot_indices=self.battery_slot,
                    )
                ),
                broadcast=["scenario", "battery_slot_indices"],
            )

        for time_step_temp in self.timesteps_minus:
            #   constraint 14)
            self.optimization_problem.define_constraint(
                (
                    "variable",
                    1,
                    dict(
                            name="battery_slot_soc",
                            timestep=time_step_temp,
                            scenario=self.scenarios,
                            battery_slot_indices=self.battery_slot,
                    )
                ),
                ">=",
                (
                    "variable",
                    big_M_constant_2*np.ones([1, int(swapping_demand.data_number_ev_to_be_swapped_dict[time_step_temp])]),
                    dict(
                            name="A_matrix",
                            timestep=time_step_temp,
                            scenario=self.scenarios,
                            battery_slot_indices=self.battery_slot,
                    )
                ),
                broadcast=["scenario", "battery_slot_indices"],
            )

        # constraint 19)
        self.optimization_problem.define_constraint(
            (
                "variable",
                1,
                dict(
                    name="battery_slot_soc",
                    timestep=self.timesteps,
                    scenario=self.scenarios,
                    battery_slot_indices=self.battery_slot,
                )
            ),
            "==",
            (
                "variable",
                1/data_set.battery_cell_data['nominal battery capacity (kWh)'].values[0],
                dict(
                    name="battery_slot_energy",
                    timestep=self.timesteps,
                    scenario=self.scenarios,
                    battery_slot_indices=self.battery_slot,
                )
            ),
            broadcast=["scenario", "timestep", "battery_slot_indices"],
        )

        # constraint 18b) set initial energy at time step 0
        self.optimization_problem.define_constraint(
            (
                "variable",
                1,
                dict(
                    name="battery_slot_energy",
                    timestep=self.timesteps[0],
                    scenario=self.scenarios,
                    battery_slot_indices=self.battery_slot,
                )
            ),
            "==",
            (
                "constant",
                data_set.battery_cell_data['nominal battery capacity (kWh)'].values[0],
            ),
            broadcast=["scenario", "battery_slot_indices"],
        )

        # constraint 20a)
        self.optimization_problem.define_constraint(
            (
                "variable",
                1,
                dict(
                        name="battery_charge_power",
                        timestep=self.timesteps_minus,
                        scenario=self.scenarios,
                        battery_slot_indices=self.battery_slot,
                )
            ),
            ">=",
            (
                "constant",
                0,
            ),
            broadcast=["scenario", "timestep", "battery_slot_indices"],
        )

        self.optimization_problem.define_constraint(
            (
                "variable",
                1,
                dict(
                        name="battery_charge_power",
                        timestep=self.timesteps_minus,
                        scenario=self.scenarios,
                        battery_slot_indices=self.battery_slot,
                )
            ),
            "<=",
            (
                "constant",
                data_set.battery_cell_data['charging power max (kW)'].values[0],
            ),
            broadcast=["scenario", "timestep", "battery_slot_indices"],
        )

        self.optimization_problem.define_constraint(
            (
                "variable",
                1,
                dict(
                        name="battery_discharge_power",
                        timestep=self.timesteps_minus,
                        scenario=self.scenarios,
                        battery_slot_indices=self.battery_slot,
                )
            ),
            ">=",
            (
                "constant",
                0,
            ),
            broadcast=["scenario", "timestep", "battery_slot_indices"],
        )

        self.optimization_problem.define_constraint(
            (
                "variable",
                1,
                dict(
                        name="battery_discharge_power",
                        timestep=self.timesteps_minus,
                        scenario=self.scenarios,
                        battery_slot_indices=self.battery_slot,
                )
            ),
            "<=",
            (
                "constant",
                data_set.battery_cell_data['charging power max (kW)'].values[0],
            ),
            broadcast=["scenario", "timestep", "battery_slot_indices"],
        )

        mesmo.utils.logger.info('Define constraints 22a)')
        # constraint 22a) total energy per swapping/charging station
        self.optimization_problem.define_constraint(
            (
                "variable",
                1,
                dict(
                        name="swapping_station_total_energy_demand",
                        scenario=self.scenarios[0],
                        timestep=self.timesteps_minus,
                )
            ),
            "==",
            (
                "variable",
                -1*np.ones([1, number_of_battery_slot]),
                dict(
                        name="battery_discharge_power",
                        scenario=self.scenarios[0],
                        timestep=self.timesteps_minus,
                )
            ),
            (
                "variable",
                np.ones([1, number_of_battery_slot]),
                dict(
                        name="battery_charge_power",
                        scenario=self.scenarios[0],
                        timestep=self.timesteps_minus,
                )
            ),
            (
                "variable",
                -1,
                dict(
                        name="contingency_reserve",
                        scenario=self.scenarios[0],
                        timestep=self.timesteps_minus,
                )
            ),
            broadcast=["timestep"],
        )

        # constraint 23a)
        self.optimization_problem.define_constraint(
            (
                "variable",
                1,
                dict(
                        name="swapping_station_total_energy_demand",
                        scenario=self.scenarios[0],
                        timestep=self.timesteps_minus,
                )
            ),
            "==",
            (
                "variable",
                -1*np.ones([1, number_of_battery_slot]),
                dict(
                        name="battery_discharge_power",
                        scenario=self.scenarios[1],
                        timestep=self.timesteps_minus,
                )
            ),
            (
                "variable",
                np.ones([1, number_of_battery_slot]),
                dict(
                        name="battery_charge_power",
                        scenario=self.scenarios[1],
                        timestep=self.timesteps_minus,
                )
            ),
            broadcast=["timestep"],
        )

        # constraint 22b)
        self.optimization_problem.define_constraint(
            (
                "variable",
                1,
                dict(
                        name="contingency_reserve",
                        timestep=self.timesteps_minus,
                        scenario=self.scenarios[0],
                )
            ),
            ">=",
            (
                "variable",
                constant_minimal_reserve, #100 kWh
                dict(
                        name="contingency_reserve_binary",
                        timestep=self.timesteps_minus,
                        scenario=self.scenarios[0],
                )
            ),
            broadcast=["timestep"],
        )

        # 23a)
        self.optimization_problem.define_constraint(
            (
                "variable",
                1,
                dict(
                        name="contingency_reserve",
                        timestep=self.timesteps_minus,
                        scenario=self.scenarios[0],
                )
            ),
            "<=",
            (
                "variable",
                big_M_constant_4,
                dict(
                        name="contingency_reserve_binary",
                        timestep=self.timesteps_minus,
                        scenario=self.scenarios[0],
                )
            ),
            broadcast=["timestep"],
        )

        # constraint final soc
        self.optimization_problem.define_constraint(
            (
                "variable",
                1,
                dict(
                    name="battery_slot_soc",
                    timestep=self.timesteps[-1],
                    scenario=self.scenarios,
                    battery_slot_indices=self.battery_slot,
                )
            ),
            ">=",
            (
                "constant",
                0.8,
            ),
            broadcast=["scenario", "battery_slot_indices"],
        )

        # not a real-constraint: evaluation purpose only
        for time_step_temp in self.timesteps_minus:
            self.optimization_problem.define_constraint(
                (
                    "variable",
                    np.ones([1, int(swapping_demand.data_number_ev_to_be_swapped_dict[time_step_temp])]),
                    dict(
                            name="A_matrix",
                            timestep=time_step_temp,
                            scenario=self.scenarios,
                            battery_slot_indices=self.battery_slot,
                            ev_indices=list(
                            range(int(swapping_demand.data_number_ev_to_be_swapped_dict[time_step_temp]))),
                    )
                ),
                "==",
                (
                    "variable",
                    1,
                    dict(
                            name="swapping_events_per_slot",
                            timestep=time_step_temp,
                            scenario=self.scenarios,
                            battery_slot_indices=self.battery_slot,
                    )
                ),
                broadcast=["scenario", "battery_slot_indices"],
            )

            score_parameter = number_of_battery_slot*swapping_demand.data_number_ev_to_be_swapped_dict[time_step_temp]

            # swapping scores calculation
            self.optimization_problem.define_constraint(
                (
                    "variable",
                    np.linspace(((score_parameter+swapping_demand.data_number_ev_to_be_swapped_dict[time_step_temp])/number_of_battery_slot),
                                1, int(score_parameter), endpoint=True),
                    dict(
                            name="A_matrix",
                            timestep=time_step_temp,
                            scenario=self.scenarios,
                            battery_slot_indices=self.battery_slot,
                            ev_indices=list(
                            range(int(swapping_demand.data_number_ev_to_be_swapped_dict[time_step_temp]))),
                    )
                ),
                "==",
                (
                    "variable",
                    1,
                    dict(
                            name="swapping_points",
                            timestep=time_step_temp,
                            scenario=self.scenarios,
                    )
                ),
                broadcast=["scenario"],
            )

        # Note: 1. trivial case if set as <=1000
        # 2. Too large value, e.g., >=8000 will cause long computation time
        # 3。 Medium-scale values leads to partial swapping of total demand. E.g., 4000 - acceptance rate of 30%
        swapping_score = 7000

        mesmo.utils.logger.info('Define objective function')
        # wep data selection
        wep = data_set.wholesale_electricity_price_data[0:self.timesteps_minus.size]["WEP ($/MWh)"]
        wep = wep.values

        temp = data_set.reserve_price_data['RESERVE GROUP'] == 'CONRESA'
        reserve_price = data_set.reserve_price_data[temp.values]["PRICE ($/MWh)"]
        reserve_price = reserve_price[0:self.timesteps_minus.size].values

        self.wep = wep
        self.reserve_price = reserve_price
        self.timestep_interval_hours = timestep_interval_hours

        self.optimization_problem.define_objective(
            (
                'variable',
                (wep * timestep_interval_hours),    # cost
                dict(name='swapping_station_total_energy_demand', timestep=self.timesteps_minus, scenario=self.scenarios[0])
            ),
            (
                'variable',
                -swapping_score*np.ones([1, self.timesteps_minus.size]),    # utility
                dict(name='swapping_points', timestep=self.timesteps_minus, scenario=self.scenarios[0])
            ),
            (
                'variable',
                -swapping_score * np.ones([1, self.timesteps_minus.size]),  # utility
                dict(name='swapping_points', timestep=self.timesteps_minus, scenario=self.scenarios[1])
            ),
            (
                'variable',
                -(reserve_price * timestep_interval_hours),  # utility
                dict(name='contingency_reserve', timestep=self.timesteps_minus, scenario=self.scenarios[0])
            ),
        )

        mesmo.utils.logger.info('BSCS contingency reserve model defined.')

# Optimisation models - energy + primary reserve model
class bscs_primary_reserve_optimization_model(object):
    def __init__(
            self,
            scenario_name,
            data_set,
            swapping_demand,
            time_step,
            enable_electric_grid_model=False
    ):
        # assign constants
        soc_threshold = 0.8
        primary_reserve_time_step_size = 2  # 2s
        market_time_step_size_in_seconds = 30 * 60  # 1800s
        big_M_constant = data_set.battery_cell_data['nominal battery capacity (kWh)'].values[
            0]  # max energy capacity per battery slot
        big_M_constant_2 = soc_threshold  # max soc = 1
        big_M_constant_3 = data_set.battery_cell_data['charging power max (kW)'].values[0]  # max charge/discharge power
        big_M_constant_4 = data_set.battery_cell_data['charging power max (kW)'].values[
            0]  # max primary reserve per slot

        mesmo.utils.logger.info('Initializing BSCS primary reserve optimisation model...')

        # Obtain DER & grid model objects.
        self.der_model_set = mesmo.der_models.DERModelSet(scenario_name)

        # settings
        # reserved for stochastic optimisation
        self.scenarios = ['deterministic']

        # battery slot index
        number_of_battery_slot = data_set.bscs_data['number_of_battery_slots'].values[0]
        battery_slot_index = list(range(number_of_battery_slot))
        self.battery_slot = ['battery_slot_no_{}'.format(x) for x in battery_slot_index]

        # total time steps
        self.timesteps = time_step
        # partial time steps
        self.timesteps_minus = time_step[0:(time_step.size - 1)]
        self.timesteps_plus = time_step[1:time_step.size]

        # Obtain timestep interval in hours, for conversion of power to energy.
        timestep_interval_hours = (
                (self.timesteps[1] - self.timesteps[0]) / pd.Timedelta('1h')
        )

        # preprocessing reg d data to synthesize primary reserve activation signal
        data_set.red_d_data.drop(data_set.red_d_data.tail(1).index, inplace=True)
        primary_reserve_timestep_interval_hours = primary_reserve_time_step_size / 3600

        # max 48 periods allowed
        cumulative_primary_reserve_duration_per_market_period = np.zeros(len(time_step))

        for data_index in range(len(data_set.red_d_data)):
            market_period_index_temp = int((data_set.red_d_data[0][data_index].hour * 3600 + data_set.red_d_data[0][
                data_index].minute * 60 + data_set.red_d_data[0][data_index].second)
                                           / market_time_step_size_in_seconds)

            if data_set.red_d_data[1][data_index] <= -0.5:
                cumulative_primary_reserve_duration_per_market_period[market_period_index_temp] \
                    += 1 * primary_reserve_timestep_interval_hours

        # Instantiate optimization problem.
        self.optimization_problem = mesmo.utils.OptimizationProblem()

        self.optimization_problem.define_variable(
            "battery_charge_power",
            scenario=self.scenarios,
            timestep=self.timesteps_minus,
            battery_slot_indices=self.battery_slot,
        )

        self.optimization_problem.define_variable(
            "battery_discharge_power",
            scenario=self.scenarios,
            timestep=self.timesteps_minus,
            battery_slot_indices=self.battery_slot,
        )

        # primary reserve variable
        self.optimization_problem.define_variable(
            "battery_charge_primary_reserve",
            scenario=self.scenarios,
            timestep=self.timesteps_minus,
            battery_slot_indices=self.battery_slot,
        )

        self.optimization_problem.define_variable(
            "battery_discharge_primary_reserve",
            scenario=self.scenarios,
            timestep=self.timesteps_minus,
            battery_slot_indices=self.battery_slot,
        )

        self.optimization_problem.define_variable(
            "battery_primary_reserve_binary",
            variable_type='binary',
            scenario=self.scenarios,
            timestep=self.timesteps_minus,
            battery_slot_indices=self.battery_slot,
        )

        self.optimization_problem.define_variable(
            'swapping_station_primary_reserve_offer',
            timestep=self.timesteps_minus,
            scenario=self.scenarios,
        )

        # bscs variables
        self.optimization_problem.define_variable(
            "battery_slot_energy",
            scenario=self.scenarios,
            timestep=self.timesteps,
            battery_slot_indices=self.battery_slot,
        )

        self.optimization_problem.define_variable(
            "battery_slot_soc",
            scenario=self.scenarios,
            timestep=self.timesteps,
            battery_slot_indices=self.battery_slot,
        )

        # Define price arbitrage variables.
        self.optimization_problem.define_variable(
            'swapping_station_total_energy_demand',
            timestep=self.timesteps_minus,
            scenario=self.scenarios,
        )

        # Define A_t matrix
        # number of incoming batteries:  swapping_demand.data_number_ev_to_be_swapped_dict[time_step[3]]
        for time_step_temp in self.timesteps_minus:
            self.optimization_problem.define_variable(
                'A_matrix',
                variable_type='binary',
                timestep=time_step_temp,
                scenario=self.scenarios,
                battery_slot_indices=self.battery_slot,
                ev_indices=list(range(int(swapping_demand.data_number_ev_to_be_swapped_dict[time_step_temp]))),
            )

            self.optimization_problem.define_variable(
                'swapping_points',
                timestep=time_step_temp,
                scenario=self.scenarios,
            )

            self.optimization_problem.define_variable(
                'swapping_events_per_slot',
                timestep=time_step_temp,
                battery_slot_indices=self.battery_slot,
                scenario=self.scenarios,
            )

            self.optimization_problem.define_variable(
                'Z_matrix',
                timestep=time_step_temp,
                scenario=self.scenarios,
                battery_slot_indices=self.battery_slot,
            )

        # Define constraints
        # Define energy constraints
        mesmo.utils.logger.info('Define energy constraints 1a)')

        i = 0
        for (timestep_temp_minus, timestep_temp_plus) in zip(self.timesteps_minus, self.timesteps_plus):
            # number of incoming EVs per timestep
            number_ev_per_step = swapping_demand.data_SOC_ev_to_be_swapped_dict[timestep_temp_minus].size
            A_coefficient_matrix = np.zeros([number_of_battery_slot, number_of_battery_slot*number_ev_per_step])
            Z_coefficient_matrix = np.zeros([number_of_battery_slot, number_of_battery_slot*number_ev_per_step])

            for index_temp in range(number_of_battery_slot):

                A_coefficient_matrix[index_temp, index_temp*number_ev_per_step: (index_temp+1)*number_ev_per_step] = \
                    swapping_demand.data_energy_ev_to_be_swapped_dict[timestep_temp_minus]

                Z_coefficient_matrix[index_temp, index_temp*number_ev_per_step: (index_temp+1)*number_ev_per_step] = -1

            A_coefficient_matrix = sp.csr_matrix(A_coefficient_matrix)
            Z_coefficient_matrix = sp.csr_matrix(Z_coefficient_matrix)

            ## Debug print('debug')
            # indices = self.optimization_problem.get_variable_index(name='Z_matrix', timestep=timestep_temp_minus)
            # var_res = self.optimization_problem.variables.iloc[indices]

            # constraint 1)
            self.optimization_problem.define_constraint(
                (
                    "variable",
                    1,
                    dict(
                            name="battery_slot_energy",
                            scenario=self.scenarios,
                            timestep=timestep_temp_plus,
                    )
                ),
                "==",
                (
                    "variable",
                    A_coefficient_matrix,
                    dict(
                            name="A_matrix",
                            scenario=self.scenarios,
                            timestep=timestep_temp_minus,
                    )
                ),
                (
                    "variable",
                    1,
                    dict(
                            name="battery_slot_energy",
                            scenario=self.scenarios,
                            timestep=timestep_temp_minus,
                    )
                ),
                (
                    "variable",
                    -1, #Z_coefficient_matrix,
                    dict(
                        name="Z_matrix",
                        scenario=self.scenarios,
                        timestep=timestep_temp_minus,
                    )
                ),
                (
                    "variable",
                    data_set.battery_cell_data['battery efficiency'].values[0] * timestep_interval_hours,
                    dict(
                            name="battery_charge_power", scenario=self.scenarios, timestep=timestep_temp_minus,
                            battery_slot_indices=self.battery_slot,
                    ),
                ),
                (
                    "variable",
                    -1/data_set.battery_cell_data['battery efficiency'].values[0] * timestep_interval_hours,
                    dict(
                            name="battery_discharge_power", scenario=self.scenarios, timestep=timestep_temp_minus,
                            battery_slot_indices=self.battery_slot
                    ),
                ),
                (
                    "variable",
                    -data_set.battery_cell_data['battery efficiency'].values[0] * cumulative_primary_reserve_duration_per_market_period[i],
                    dict(
                        name="battery_charge_primary_reserve", scenario=self.scenarios, timestep=timestep_temp_minus,
                        battery_slot_indices=self.battery_slot,
                    ),
                ),
                (
                    "variable",
                    -1 / data_set.battery_cell_data['battery efficiency'].values[0] * cumulative_primary_reserve_duration_per_market_period[i],
                    dict(
                        name="battery_discharge_primary_reserve", scenario=self.scenarios, timestep=timestep_temp_minus,
                        battery_slot_indices=self.battery_slot
                    ),
                ),
                broadcast=["scenario"],
            )

        i += 1


        mesmo.utils.logger.info('Define constraints 1b) - 1e)')
        # constraint 9a)
        self.optimization_problem.define_constraint(
            (
                "variable",
                1,
                dict(
                        name="battery_slot_energy",
                        scenario=self.scenarios,
                        timestep=self.timesteps,
                        battery_slot_indices=self.battery_slot,
                )
            ),
            "<=",
            (
                "constant",
                data_set.battery_cell_data["max battery capacity (kWh)"].values[0],
                dict(
                        scenario=self.scenarios,
                        timestep=self.timesteps,
                        battery_slot_indices=self.battery_slot,
                )
            ),
            broadcast=["scenario", "timestep", "battery_slot_indices"],
        )

        # constraint 9b) part 1
        for time_step_temp in self.timesteps_minus:
            self.optimization_problem.define_constraint(
                (
                    "variable",
                    1,
                    dict(
                        name="Z_matrix",
                        timestep=time_step_temp,
                        scenario=self.scenarios,
                        battery_slot_indices=self.battery_slot,
                    )
                ),
                ">=",
                (
                    "constant",
                    0,
                ),
                broadcast=["scenario", "battery_slot_indices"],
            )

        # constraint 9b) part 2
        for time_step_temp in self.timesteps_minus:
            self.optimization_problem.define_constraint(
                (
                    "variable",
                    1,
                    dict(
                        name="Z_matrix",
                        timestep=time_step_temp,
                        scenario=self.scenarios,
                        battery_slot_indices=self.battery_slot,
                    )
                ),
                "<=",
                (
                    "variable",
                    1,
                    dict(
                        name="battery_slot_energy",
                        timestep=time_step_temp,
                        scenario=self.scenarios,
                        battery_slot_indices=self.battery_slot,
                    )
                ),
                broadcast=["scenario", "battery_slot_indices"],
            )

        # Define constraint 9c)
        for time_step_temp in self.timesteps_minus:
            self.optimization_problem.define_constraint(
                (
                    "variable",
                    1,
                    dict(
                        name="Z_matrix",
                        timestep=time_step_temp,
                        scenario=self.scenarios,
                        battery_slot_indices=self.battery_slot
                    )
                ),
                "<=",
                (
                    "variable",
                    big_M_constant*np.ones([1, int(swapping_demand.data_number_ev_to_be_swapped_dict[time_step_temp])]),
                    dict(
                        name="A_matrix",
                        timestep=time_step_temp,
                        scenario=self.scenarios,
                        battery_slot_indices=self.battery_slot,
                    )
                ),
                broadcast=["scenario", "battery_slot_indices"],
            )

        # Define 9d)
        for time_step_temp in self.timesteps_minus:
            self.optimization_problem.define_constraint(
                (
                    "variable",
                    1,
                    dict(
                            name="Z_matrix",
                            timestep=time_step_temp,
                            scenario=self.scenarios,
                            battery_slot_indices=self.battery_slot,
                    )
                ),
                ">=",
                (
                    "variable",
                    1,
                    dict(
                            name="battery_slot_energy",
                            timestep=time_step_temp,
                            scenario=self.scenarios,
                            battery_slot_indices=self.battery_slot,
                    )
                ),
                (
                    "constant",
                    -big_M_constant,
                    dict(
                            scenario=self.scenarios,
                            timestep=time_step_temp,
                            battery_slot_indices=self.battery_slot,
                    )
                ),
                (
                    "variable",
                    big_M_constant*np.ones([1, int(swapping_demand.data_number_ev_to_be_swapped_dict[time_step_temp])]),
                    dict(
                            name="A_matrix",
                            timestep=time_step_temp,
                            scenario=self.scenarios,
                            battery_slot_indices=self.battery_slot,
                    )
                ),
                broadcast=["scenario", "battery_slot_indices"],
            )

        mesmo.utils.logger.info('Define constraints 2a) - 2e)')
        # constraint 10)
        for time_step_temp in self.timesteps_minus:
            self.optimization_problem.define_constraint(
                (
                    "variable",
                    np.ones([1, number_of_battery_slot]),
                    dict(
                        name="A_matrix",
                        timestep=time_step_temp,
                        scenario=self.scenarios,
                        battery_slot_indices=self.battery_slot,
                        ev_indices=list(range(int(swapping_demand.data_number_ev_to_be_swapped_dict[time_step_temp]))),
                    )
                ),
                "<=",
                (
                    "constant",
                    1,
                    dict(
                        timestep=time_step_temp,
                        scenario=self.scenarios,
                        battery_slot_indices=self.battery_slot,
                        ev_indices=list(range(int(swapping_demand.data_number_ev_to_be_swapped_dict[time_step_temp]))),
                    )
                ),
                broadcast=["scenario", "ev_indices"],
            )

        for time_step_temp in self.timesteps_minus:
            self.optimization_problem.define_constraint(
                (
                    "variable",
                    np.ones([1, int(swapping_demand.data_number_ev_to_be_swapped_dict[time_step_temp])]),
                    dict(
                            name="A_matrix",
                            timestep=time_step_temp,
                            scenario=self.scenarios,
                            battery_slot_indices=self.battery_slot,
                    )
                ),
                ">=",
                (
                    "constant",
                    0,
                    dict(
                            timestep=time_step_temp,
                            scenario=self.scenarios,
                            battery_slot_indices=self.battery_slot,
                    )
                ),
                broadcast=["scenario", "battery_slot_indices"],
            )

            # Debug
            # indices = self.optimization_problem.get_variable_index(name='Z_matrix', timestep=timestep_temp_minus)
            # var_res = self.optimization_problem.variables.iloc[indices]

        for time_step_temp in self.timesteps_minus:
            #   constraint 14)
            self.optimization_problem.define_constraint(
                (
                    "variable",
                    1,
                    dict(
                            name="battery_slot_soc",
                            timestep=time_step_temp,
                            scenario=self.scenarios,
                            battery_slot_indices=self.battery_slot,
                    )
                ),
                ">=",
                (
                    "variable",
                    big_M_constant_2*np.ones([1, int(swapping_demand.data_number_ev_to_be_swapped_dict[time_step_temp])]),
                    dict(
                            name="A_matrix",
                            timestep=time_step_temp,
                            scenario=self.scenarios,
                            battery_slot_indices=self.battery_slot,
                    )
                ),
                broadcast=["scenario", "battery_slot_indices"],
            )

        # constraint 19)
        self.optimization_problem.define_constraint(
            (
                "variable",
                1,
                dict(
                    name="battery_slot_soc",
                    timestep=self.timesteps,
                    scenario=self.scenarios,
                    battery_slot_indices=self.battery_slot,
                )
            ),
            "==",
            (
                "variable",
                1/data_set.battery_cell_data['nominal battery capacity (kWh)'].values[0],
                dict(
                    name="battery_slot_energy",
                    timestep=self.timesteps,
                    scenario=self.scenarios,
                    battery_slot_indices=self.battery_slot,
                )
            ),
            broadcast=["scenario", "timestep", "battery_slot_indices"],
        )

        # constraint 18b) set initial energy at time step 0
        self.optimization_problem.define_constraint(
            (
                "variable",
                1,
                dict(
                    name="battery_slot_energy",
                    timestep=self.timesteps[0],
                    scenario=self.scenarios,
                    battery_slot_indices=self.battery_slot,
                )
            ),
            "==",
            (
                "constant",
                data_set.battery_cell_data['nominal battery capacity (kWh)'].values[0],
            ),
            broadcast=["scenario", "battery_slot_indices"],
        )

        # constraint 20a)
        self.optimization_problem.define_constraint(
            (
                "variable",
                1,
                dict(
                    name="battery_charge_power",
                    timestep=self.timesteps_minus,
                    scenario=self.scenarios,
                    battery_slot_indices=self.battery_slot,
                )
            ),
            ">=",
            (
                "constant",
                0,
            ),
            broadcast=["scenario", "timestep", "battery_slot_indices"],
        )

        self.optimization_problem.define_constraint(
            (
                "variable",
                1,
                dict(
                    name="battery_charge_power",
                    timestep=self.timesteps_minus,
                    scenario=self.scenarios,
                    battery_slot_indices=self.battery_slot,
                )
            ),
            "<=",
            (
                "constant",
                data_set.battery_cell_data['charging power max (kW)'].values[0],
            ),
            broadcast=["scenario", "timestep", "battery_slot_indices"],
        )

        self.optimization_problem.define_constraint(
            (
                "variable",
                1,
                dict(
                    name="battery_discharge_power",
                    timestep=self.timesteps_minus,
                    scenario=self.scenarios,
                    battery_slot_indices=self.battery_slot,
                )
            ),
            ">=",
            (
                "constant",
                0,
            ),
            broadcast=["scenario", "timestep", "battery_slot_indices"],
        )

        self.optimization_problem.define_constraint(
            (
                "variable",
                1,
                dict(
                    name="battery_discharge_power",
                    timestep=self.timesteps_minus,
                    scenario=self.scenarios,
                    battery_slot_indices=self.battery_slot,
                )
            ),
            "<=",
            (
                "constant",
                data_set.battery_cell_data['charging power max (kW)'].values[0],
            ),
            broadcast=["scenario", "timestep", "battery_slot_indices"],
        )

        # constraint final soc
        self.optimization_problem.define_constraint(
            (
                "variable",
                1,
                dict(
                    name="battery_slot_soc",
                    timestep=self.timesteps[-1],
                    scenario=self.scenarios,
                    battery_slot_indices=self.battery_slot,
                )
            ),
            ">=",
            (
                "constant",
                0.1,
            ),
            broadcast=["scenario", "battery_slot_indices"],
        )

        i = 0
        #  constraint 10
        for time_step_temp in self.timesteps_minus:
            self.optimization_problem.define_constraint(
                (
                    "variable",
                    np.ones([1, int(swapping_demand.data_number_ev_to_be_swapped_dict[time_step_temp])]),
                    dict(
                            name="A_matrix",
                            timestep=time_step_temp,
                            scenario=self.scenarios,
                            battery_slot_indices=self.battery_slot,
                            ev_indices=list(
                            range(int(swapping_demand.data_number_ev_to_be_swapped_dict[time_step_temp]))),
                    )
                ),
                "==",
                (
                    "variable",
                    1,
                    dict(
                            name="swapping_events_per_slot",
                            timestep=time_step_temp,
                            scenario=self.scenarios,
                            battery_slot_indices=self.battery_slot,
                    )
                ),
                broadcast=["scenario", "battery_slot_indices"],
            )

            score_parameter = number_of_battery_slot*swapping_demand.data_number_ev_to_be_swapped_dict[time_step_temp]

            # magic function to set the start parameter in linspace(), it seems to impact the solution time a lot
            # (score_parameter+1)/13 seems to work well if rand(1,3) for incoming evs
            # (score_parameter + swapping_demand.data_number_ev_to_be_swapped_dict[
            #    time_step_temp]) / number_of_battery_slot seems to work well if rand(1,3) for incoming evs

            i += 0
            # swapping scores calculation
            self.optimization_problem.define_constraint(
                (
                    "variable",
                    np.linspace(i+((score_parameter+swapping_demand.data_number_ev_to_be_swapped_dict[time_step_temp])/number_of_battery_slot),
                                i+1, int(score_parameter), endpoint=True),
                    dict(
                            name="A_matrix",
                            timestep=time_step_temp,
                            scenario=self.scenarios,
                            battery_slot_indices=self.battery_slot,
                            ev_indices=list(
                            range(int(swapping_demand.data_number_ev_to_be_swapped_dict[time_step_temp]))),
                    )
                ),
                "==",
                (
                    "variable",
                    1,
                    dict(
                            name="swapping_points",
                            timestep=time_step_temp,
                            scenario=self.scenarios,
                    )
                ),
            )

        # constraint 11
        # a)
        for time_step_temp in self.timesteps_minus:
            self.optimization_problem.define_constraint(
                (
                    "variable",
                    1,
                    dict(
                        name="battery_charge_power",
                        timestep=time_step_temp,
                        scenario=self.scenarios,
                        battery_slot_indices=self.battery_slot,
                    )
                ),
                "<=",
                (
                    "constant",
                    big_M_constant_3,
                ),
                (
                    "variable",
                    -big_M_constant_3*np.ones([1, int(swapping_demand.data_number_ev_to_be_swapped_dict[time_step_temp])]),
                    dict(
                            name="A_matrix",
                            timestep=time_step_temp,
                            scenario=self.scenarios,
                            battery_slot_indices=self.battery_slot,
                    )
                ),
                broadcast=["scenario", "battery_slot_indices"],
            )

            # b)
            self.optimization_problem.define_constraint(
                (
                    "variable",
                    1,
                    dict(
                        name="battery_discharge_power",
                        timestep=time_step_temp,
                        scenario=self.scenarios,
                        battery_slot_indices=self.battery_slot,
                    )
                ),
                "<=",
                (
                    "constant",
                    big_M_constant_3,
                ),
                (
                    "variable",
                    -big_M_constant_3*np.ones([1, int(swapping_demand.data_number_ev_to_be_swapped_dict[time_step_temp])]),
                    dict(
                            name="A_matrix",
                            timestep=time_step_temp,
                            scenario=self.scenarios,
                            battery_slot_indices=self.battery_slot,
                    )
                ),
                broadcast=["scenario", "battery_slot_indices"],
            )

        ## primary reserve constraints:
        # self.optimization_problem.define_constraint(
        #     (
        #         "variable",
        #         1,
        #         dict(
        #                 name="battery_charge_primary_reserve",
        #                 timestep=self.timesteps_minus,
        #                 scenario=self.scenarios,
        #                 battery_slot_indices=self.battery_slot,
        #         )
        #     ),
        #     "==",
        #     (
        #         "constant",
        #         0,
        #     ),
        #     broadcast=["scenario", "timestep", "battery_slot_indices"],
        # )
        #
        # self.optimization_problem.define_constraint(
        #     (
        #         "variable",
        #         1,
        #         dict(
        #                 name="battery_discharge_primary_reserve",
        #                 timestep=self.timesteps_minus,
        #                 scenario=self.scenarios,
        #                 battery_slot_indices=self.battery_slot,
        #         )
        #     ),
        #     "==",
        #     (
        #         "constant",
        #         0,
        #     ),
        #     broadcast=["scenario", "timestep", "battery_slot_indices"],
        # )
        #
        # self.optimization_problem.define_constraint(
        #     (
        #         "variable",
        #         1,
        #         dict(
        #                 name="swapping_station_primary_reserve_offer",
        #                 timestep=self.timesteps_minus,
        #                 scenario=self.scenarios,
        #         )
        #     ),
        #     "==",
        #     (
        #         "constant",
        #         0,
        #     ),
        #     broadcast=["scenario", "timestep"],
        # )
        # constraint 32a-b)
        for time_step_temp in self.timesteps_minus:
            self.optimization_problem.define_constraint(
                (
                    "variable",
                    1,
                    dict(
                            name="battery_charge_primary_reserve",
                            timestep=time_step_temp,
                            scenario=self.scenarios,
                            battery_slot_indices=self.battery_slot,
                    )
                ),
                "<=",
                (
                    "constant",
                    big_M_constant_4,
                ),
                (
                    "variable",
                    -big_M_constant_4 * np.ones(
                        [1, int(swapping_demand.data_number_ev_to_be_swapped_dict[time_step_temp])]),
                    dict(
                            name="A_matrix",
                            timestep=time_step_temp,
                            scenario=self.scenarios,
                            battery_slot_indices=self.battery_slot,
                    )
                ),
                broadcast=["scenario", "battery_slot_indices"],
            )

            self.optimization_problem.define_constraint(
                (
                    "variable",
                    1,
                    dict(
                            name="battery_discharge_primary_reserve",
                            timestep=time_step_temp,
                            scenario=self.scenarios,
                            battery_slot_indices=self.battery_slot,
                    )
                ),
                "<=",
                (
                    "constant",
                    big_M_constant_4,
                ),
                (
                    "variable",
                    -big_M_constant_4 * np.ones(
                        [1, int(swapping_demand.data_number_ev_to_be_swapped_dict[time_step_temp])]),
                    dict(
                            name="A_matrix",
                            timestep=time_step_temp,
                            scenario=self.scenarios,
                            battery_slot_indices=self.battery_slot,
                    )
                ),
                broadcast=["scenario", "battery_slot_indices"],
            )

        self.optimization_problem.define_constraint(
            (
                "variable",
                1,
                dict(
                        name="battery_charge_primary_reserve",
                        timestep=self.timesteps_minus,
                        scenario=self.scenarios,
                        battery_slot_indices=self.battery_slot,
                )
            ),
            ">=",
            (
                "constant",
                0,
            ),
            broadcast=["scenario", "timestep", "battery_slot_indices"],
        )

        self.optimization_problem.define_constraint(
            (
                "variable",
                1,
                dict(
                        name="battery_discharge_primary_reserve",
                        timestep=self.timesteps_minus,
                        scenario=self.scenarios,
                        battery_slot_indices=self.battery_slot,
                )
            ),
            ">=",
            (
                "constant",
                0,
            ),
            broadcast=["scenario", "timestep", "battery_slot_indices"],
        )

        # constraint 33a-b)
        self.optimization_problem.define_constraint(
            (
                "variable",
                1,
                dict(
                        name="battery_charge_primary_reserve",
                        timestep=self.timesteps_minus,
                        scenario=self.scenarios,
                        battery_slot_indices=self.battery_slot,
                )
            ),
            "<=",
            (
                "variable",
                1,
                dict(
                        name="battery_charge_power",
                        timestep=self.timesteps_minus,
                        scenario=self.scenarios,
                        battery_slot_indices=self.battery_slot,
                )
            ),
            broadcast=["scenario", "timestep", "battery_slot_indices"],
        )

        self.optimization_problem.define_constraint(
            (
                "variable",
                1,
                dict(
                        name="battery_discharge_primary_reserve",
                        timestep=self.timesteps_minus,
                        scenario=self.scenarios,
                        battery_slot_indices=self.battery_slot,
                )
            ),
            "<=",
            (
                "variable",
                1,
                dict(
                        name="battery_discharge_power",
                        timestep=self.timesteps_minus,
                        scenario=self.scenarios,
                        battery_slot_indices=self.battery_slot,
                )
            ),
            broadcast=["scenario", "timestep", "battery_slot_indices"],
        )

        self.optimization_problem.define_constraint(
            (
                "variable",
                1,
                dict(
                        name="battery_charge_power",
                        timestep=self.timesteps_minus,
                        scenario=self.scenarios,
                        battery_slot_indices=self.battery_slot,
                )
            ),
            "<=",
            (
                "variable",
                big_M_constant_4,
                dict(
                        name="battery_primary_reserve_binary",
                        timestep=self.timesteps_minus,
                        scenario=self.scenarios,
                        battery_slot_indices=self.battery_slot,
                )
            ),
            broadcast=["scenario", "timestep", "battery_slot_indices"],
        )

        self.optimization_problem.define_constraint(
            (
                "variable",
                1,
                dict(
                        name="battery_discharge_power",
                        timestep=self.timesteps_minus,
                        scenario=self.scenarios,
                        battery_slot_indices=self.battery_slot,
                )
            ),
            "<=",
            (
                "constant",
                big_M_constant_4,
            ),
            (
                "variable",
                -big_M_constant_4,
                dict(
                        name="battery_primary_reserve_binary",
                        timestep=self.timesteps_minus,
                        scenario=self.scenarios,
                        battery_slot_indices=self.battery_slot,
                )
            ),
            broadcast=["scenario", "timestep", "battery_slot_indices"],
        )

        # constraint 29)
        self.optimization_problem.define_constraint(
            (
                "variable",
                1,
                dict(
                        name="swapping_station_primary_reserve_offer",
                        scenario=self.scenarios,
                        timestep=self.timesteps_minus,
                )
            ),
            "==",
            (
                "variable",
                np.ones([1, number_of_battery_slot]),
                dict(
                        name="battery_charge_primary_reserve",
                        scenario=self.scenarios,
                        timestep=self.timesteps_minus,
                )
            ),
            (
                "variable",
                np.ones([1, number_of_battery_slot]),
                dict(
                        name="battery_discharge_primary_reserve",
                        scenario=self.scenarios,
                        timestep=self.timesteps_minus,
                )
            ),
            broadcast=["scenario", "timestep"],
        )


        mesmo.utils.logger.info('Define constraints 20b)')
        # constraint 8 total energy per swapping/charging station
        n = 0
        for time_step_temp in self.timesteps_minus:
            self.optimization_problem.define_constraint(
                (
                    "variable",
                    1,
                    dict(
                            name="swapping_station_total_energy_demand",
                            scenario=self.scenarios,
                            timestep=time_step_temp,
                    )
                ),
                "==",
                (
                    "variable",
                    -1*np.ones([1, number_of_battery_slot]),
                    dict(
                            name="battery_discharge_power",
                            scenario=self.scenarios,
                            timestep=time_step_temp,
                    )
                ),
                (
                    "variable",
                    np.ones([1, number_of_battery_slot]),
                    dict(
                            name="battery_charge_power",
                            scenario=self.scenarios,
                            timestep=time_step_temp,
                    )
                ),
                (
                    "variable",
                    -cumulative_primary_reserve_duration_per_market_period[n],
                    dict(
                            name="swapping_station_primary_reserve_offer",
                            scenario=self.scenarios,
                            timestep=time_step_temp,
                    )
                ),
                broadcast=["scenario"],
            )

            n += 1


        # Note: 1. trivial case if set as <=1000
        # 2. Too large value, e.g., >=8000 will cause long computation time
        # 3。 Medium-scale values leads to partial swapping of total demand. E.g., 4000 - acceptance rate of 30%
        swapping_score = 7000

        mesmo.utils.logger.info('Define objective function')
        # wep data selection
        wep = data_set.wholesale_electricity_price_data[0:self.timesteps_minus.size]["WEP ($/MWh)"]
        wep = wep.values

        temp = data_set.reserve_price_data['RESERVE GROUP'] == 'PRIRESA'
        reserve_price = data_set.reserve_price_data[temp.values]["PRICE ($/MWh)"]
        reserve_price = reserve_price[0:self.timesteps_minus.size].values

        self.wep = wep
        self.reserve_price = reserve_price
        self.timestep_interval_hours = timestep_interval_hours

        self.optimization_problem.define_objective(
            (
                'variable',
                (wep * timestep_interval_hours),
                dict(name='swapping_station_total_energy_demand', timestep=self.timesteps_minus)
            ),
            (
                'variable',
                -swapping_score*np.ones([1, self.timesteps_minus.size]),
                dict(name='swapping_points', timestep=self.timesteps_minus)
            ),
            (
                'variable',
                -(reserve_price * timestep_interval_hours),
                dict(name='swapping_station_primary_reserve_offer', timestep=self.timesteps_minus)
            ),
        )

        mesmo.utils.logger.info('BSCS WEP model defined!')

def main():
    ...


if __name__ == '__main__':
    main()
