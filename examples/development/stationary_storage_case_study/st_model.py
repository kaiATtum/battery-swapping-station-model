import numpy as np
import pandas as pd
import scipy.sparse as sp
import mesmo
import datetime
import sys
import plotly.graph_objects as go

# Optimisation models - energy model
class stationary_storage_wep_optimization_model(object):
    def __init__(
            self,
            scenario_name,
            data_set
    ):

        mesmo.utils.logger.info('Initializing stationary storage wholesale market optimisation model...')

        # define constants
        time_steps_half_hour = np.unique(np.datetime_as_string(data_set.time_stamp_half_hour))
        time_steps_day = np.unique(np.datetime_as_string(data_set.time_stamp_daily))

        # for SOC/offer constraints
        self.timesteps_minus_half_hour = time_steps_half_hour[0:(len(time_steps_half_hour)-1)]
        self.timesteps_plus_half_hour = time_steps_half_hour[1:len(time_steps_half_hour)]

        self.timesteps_minus_day = time_steps_day[0:(len(time_steps_day)-1)]
        self.timesteps_plus_day = time_steps_day[1:len(time_steps_day)]

        # unit: hour
        delta_D = 24
        delta_T = 0.5

        battery_capacity = data_set.battery_data.query('parameter == "Max storage volume"' )["Values"].values[0]
        battery_charge_efficiency = 1- data_set.battery_data.query('parameter == "Battery charging efficiency"')["Values"].values[0]
        battery_discharge_efficiency = 1 - data_set.battery_data.query('parameter == "Battery discharging efficiency"')["Values"].values[0]
        capex = data_set.battery_data.query('parameter == "Capex"')["Values"].values[0]
        degradation_rate = data_set.battery_data.query('parameter == "Storage volume degradation rate"')["Values"].values[0]
        life_cycles = data_set.battery_data.query('parameter == "Lifetime (2)"')["Values"].values[0]
        max_charge_rate = data_set.battery_data.query('parameter == "Max charging rate"')["Values"].values[0]
        max_discharge_rate = data_set.battery_data.query('parameter == "Max discharging rate"')["Values"].values[0]

        # degradation cost, unit: GBP/MW
        degradation_cost_coefficient = delta_T/battery_capacity*capex/life_cycles*degradation_rate*100
        # regulator coefficient
        phi = 1e-1

        # extract market price
        market_1_price =data_set.wholesale_price_data_half_hour["Market 1 Price [£/MWh]"].values
        market_2_price = data_set.wholesale_price_data_half_hour["Market 2 Price [£/MWh]"].values
        market_3_price = data_set.wholesale_price_data_daily["Market 3 Price [£/MWh]"].values

        # plot price, check irregularities
        fig = go.Figure()

        fig.add_scatter(x=np.arange(len(data_set.time_stamp_half_hour)),  # hour
                        y=market_1_price,
                        name='market_1',  # kW
                        ).update_traces(mode="lines")

        fig.add_scatter(x=np.arange(len(data_set.time_stamp_half_hour)),  # hour
                        y=market_2_price,
                        name='market_2',  # kW
                        ).update_traces(mode="lines")

        fig.show()

        fig = go.Figure()

        fig.add_scatter(x=np.arange(len(data_set.time_stamp_daily)),  # hour
                        y=market_3_price,
                        name='market_3',  # kW
                        ).update_traces(mode="lines")

        fig.show()

        # Instantiate optimization problem.
        self.optimization_problem = mesmo.utils.OptimizationProblem()

        # deterministic problem
        self.scenarios = ['deterministic']

        # define variables
        self.optimization_problem.define_variable(
            "battery_charge_power",
            scenario=self.scenarios,
            timestep=self.timesteps_minus_half_hour,
        )

        ## Debug: missing 6 variables for battery_discharge_power
        # self.optimization_problem.variables["timestep"] some missing values & discontonity
        # compare: self.optimization_problem.variables["timestep"].values.tolist()  and self.timesteps_minus_half_hour
        # len(np.unique(np.datetime_as_string(data_set.time_stamp_half_hour))) = 52602 some redudant values
        # total half-hour  time steps number: (365*3+1)*48

        # debug: np.ones([1,len(self.timesteps_minus_half_hour)]).shape
        # len(self.optimization_problem.get_variable_index(name='battery_charge_power'))

        self.optimization_problem.define_variable(
            "battery_discharge_power",
            scenario=self.scenarios,
            timestep=self.timesteps_minus_half_hour,
        )

        self.optimization_problem.define_variable(
            "binary_charge_discharge",
            variable_type='binary',
            scenario=self.scenarios,
            timestep=self.timesteps_minus_half_hour,
        )

        self.optimization_problem.define_variable(
            "import_power_market_1",
            scenario=self.scenarios,
            timestep=self.timesteps_minus_half_hour,
        )

        self.optimization_problem.define_variable(
            "import_power_market_2",
            scenario=self.scenarios,
            timestep=self.timesteps_minus_half_hour,
        )

        self.optimization_problem.define_variable(
            "import_power_market_3",
            scenario=self.scenarios,
            timestep=self.timesteps_minus_half_hour,
        )

        self.optimization_problem.define_variable(
            "export_power_market_1",
            scenario=self.scenarios,
            timestep=self.timesteps_minus_half_hour,
        )

        self.optimization_problem.define_variable(
            "export_power_market_2",
            scenario=self.scenarios,
            timestep=self.timesteps_minus_half_hour,
        )

        self.optimization_problem.define_variable(
            "export_power_market_3",
            scenario=self.scenarios,
            timestep=self.timesteps_minus_half_hour,
        )

        self.optimization_problem.define_variable(
            "soc",
            scenario=self.scenarios,
            timestep=time_steps_half_hour,
        )

        # Define constraints
        ## Debug print('debug')
        # indices = self.optimization_problem.get_variable_index(name='Z_matrix', timestep=timestep_temp_minus)
        # var_res = self.optimization_problem.variables.iloc[indices]

        # constraints 1b) - 1c)
        self.optimization_problem.define_constraint(
            (
                "variable",
                1,
                dict(
                    name="import_power_market_1",
                    timestep=self.timesteps_minus_half_hour,
                    scenario=self.scenarios,
                )
            ),
            (
                "variable",
                1,
                dict(
                    name="import_power_market_2",
                    timestep=self.timesteps_minus_half_hour,
                    scenario=self.scenarios,
                )
            ),
            (
                "variable",
                1,
                dict(
                    name="import_power_market_3",
                    timestep=self.timesteps_minus_half_hour,
                    scenario=self.scenarios,
                )
            ),
            "==",
            (
                "variable",
                1,
                dict(
                    name="battery_charge_power",
                    timestep=self.timesteps_minus_half_hour,
                    scenario=self.scenarios,
                )
            ),
            broadcast=["scenario","timestep"],
        )

        self.optimization_problem.define_constraint(
            (
                "variable",
                1,
                dict(
                    name="export_power_market_1",
                    timestep=self.timesteps_minus_half_hour,
                    scenario=self.scenarios,
                )
            ),
            (
                "variable",
                1,
                dict(
                    name="export_power_market_2",
                    timestep=self.timesteps_minus_half_hour,
                    scenario=self.scenarios,
                )
            ),
            (
                "variable",
                1,
                dict(
                    name="export_power_market_3",
                    timestep=self.timesteps_minus_half_hour,
                    scenario=self.scenarios,
                )
            ),
            "==",
            (
                "variable",
                1,
                dict(
                    name="battery_discharge_power",
                    timestep=self.timesteps_minus_half_hour,
                    scenario=self.scenarios,
                )
            ),
            broadcast=["scenario", "timestep"],
        )

        # for calculating the loop progress
        i = 0

        for day_step in self.timesteps_minus_day:

            sys.stdout.write(
                "\r%d%% of loop constraints extracted - to be repalced by broadcast in the future" % int(i / len(self.timesteps_minus_day) * 100 + 1)
            )
            sys.stdout.flush()
            i += 1

            # find the corresponding hourly steps:
            index_temp = np.where(self.timesteps_minus_half_hour == day_step)[0][0]
            half_hour_steps_selected_minus = self.timesteps_minus_half_hour[index_temp:index_temp+47]
            half_hour_steps_selected_plus = self.timesteps_minus_half_hour[index_temp+1:index_temp + 48]

            self.optimization_problem.define_constraint(
                (
                    "variable",
                    1,
                    dict(
                        name="import_power_market_3",
                        timestep=half_hour_steps_selected_minus,
                        scenario=self.scenarios,
                    )
                ),
                "==",
                (
                    "variable",
                    1,
                    dict(
                        name="import_power_market_3",
                        timestep=half_hour_steps_selected_plus,
                        scenario=self.scenarios,
                    )
                ),
                broadcast=["scenario"],
            )

            self.optimization_problem.define_constraint(
                (
                    "variable",
                    1,
                    dict(
                        name="export_power_market_3",
                        timestep=half_hour_steps_selected_minus,
                        scenario=self.scenarios,
                    )
                ),
                "==",
                (
                    "variable",
                    1,
                    dict(
                        name="export_power_market_3",
                        timestep=half_hour_steps_selected_plus,
                        scenario=self.scenarios,
                    )
                ),
                broadcast=["scenario"],
            )

        # constraints 1d)- 1e)
        self.optimization_problem.define_constraint(
            (
                "variable",
                1,
                dict(
                    name="battery_charge_power",
                    timestep=self.timesteps_minus_half_hour,
                    scenario=self.scenarios,
                )
            ),
            ">=",
            (
                "constant",
                0,
            ),
            broadcast=["scenario", "timestep"],
        )

        self.optimization_problem.define_constraint(
            (
                "variable",
                1,
                dict(
                    name="battery_discharge_power",
                    timestep=self.timesteps_minus_half_hour,
                    scenario=self.scenarios,
                )
            ),
            ">=",
            (
                "constant",
                0,
            ),
            broadcast=["scenario", "timestep"],
        )

        # self.optimization_problem.define_constraint(
        #     (
        #         "variable",
        #         1,
        #         dict(
        #             name="battery_charge_power",
        #             timestep=self.timesteps_minus_half_hour,
        #             scenario=self.scenarios,
        #         )
        #     ),
        #     "<=",
        #     (
        #         "constant",
        #         max_charge_rate,
        #     ),
        #     broadcast=["scenario", "timestep"],
        # )
        #
        # self.optimization_problem.define_constraint(
        #     (
        #         "variable",
        #         1,
        #         dict(
        #             name="battery_discharge_power",
        #             timestep=self.timesteps_minus_half_hour,
        #             scenario=self.scenarios,
        #         )
        #     ),
        #     "<=",
        #     (
        #         "constant",
        #         max_discharge_rate,
        #     ),
        #     broadcast=["scenario", "timestep"],
        # )
        self.optimization_problem.define_constraint(
            (
                "variable",
                1,
                dict(
                    name="battery_charge_power",
                    timestep=self.timesteps_minus_half_hour,
                    scenario=self.scenarios,
                )
            ),
            "<=",
            (
                "variable",
                max_charge_rate,
                dict(
                    name="binary_charge_discharge",
                    timestep=self.timesteps_minus_half_hour,
                    scenario=self.scenarios,
                )
            ),
            broadcast=["scenario", "timestep"],
        )

        self.optimization_problem.define_constraint(
            (
                "variable",
                1,
                dict(
                    name="battery_discharge_power",
                    timestep=self.timesteps_minus_half_hour,
                    scenario=self.scenarios,
                )
            ),
            "<=",
            (
                "constant",
                max_discharge_rate,
            ),
            (
                "variable",
                -max_discharge_rate,
                dict(
                    name="binary_charge_discharge",
                    timestep=self.timesteps_minus_half_hour,
                    scenario=self.scenarios,
                )
            ),
            broadcast=["scenario", "timestep"],
        )

        # constraints 1f
        self.optimization_problem.define_constraint(
            (
                "variable",
                1,
                dict(
                    name="import_power_market_1",
                    timestep=self.timesteps_minus_half_hour,
                    scenario=self.scenarios,
                )
            ),
            ">=",
            (
                "constant",
                0,
            ),
            broadcast=["scenario", "timestep"],
        )

        self.optimization_problem.define_constraint(
            (
                "variable",
                1,
                dict(
                    name="export_power_market_1",
                    timestep=self.timesteps_minus_half_hour,
                    scenario=self.scenarios,
                )
            ),
            ">=",
            (
                "constant",
                0,
            ),
            broadcast=["scenario", "timestep"],
        )

        self.optimization_problem.define_constraint(
            (
                "variable",
                1,
                dict(
                    name="import_power_market_2",
                    timestep=self.timesteps_minus_half_hour,
                    scenario=self.scenarios,
                )
            ),
            ">=",
            (
                "constant",
                0,
            ),
            broadcast=["scenario", "timestep"],
        )

        self.optimization_problem.define_constraint(
            (
                "variable",
                1,
                dict(
                    name="export_power_market_2",
                    timestep=self.timesteps_minus_half_hour,
                    scenario=self.scenarios,
                )
            ),
            ">=",
            (
                "constant",
                0,
            ),
            broadcast=["scenario", "timestep"],
        )

        self.optimization_problem.define_constraint(
            (
                "variable",
                1,
                dict(
                    name="import_power_market_3",
                    timestep=self.timesteps_minus_half_hour,
                    scenario=self.scenarios,
                )
            ),
            ">=",
            (
                "constant",
                0,
            ),
            broadcast=["scenario", "timestep"],
        )

        self.optimization_problem.define_constraint(
            (
                "variable",
                1,
                dict(
                    name="export_power_market_3",
                    timestep=self.timesteps_minus_half_hour,
                    scenario=self.scenarios,
                )
            ),
            ">=",
            (
                "constant",
                0,
            ),
            broadcast=["scenario", "timestep"],
        )

        # constraint 1g) -1h)
        self.optimization_problem.define_constraint(
            (
                "variable",
                1,
                dict(
                    name="soc",
                    timestep=self.timesteps_plus_half_hour,
                    scenario=self.scenarios,
                )
            ),
            "==",
            (
                "variable",
                1,
                dict(
                    name="soc",
                    timestep=self.timesteps_minus_half_hour,
                    scenario=self.scenarios,
                )
            ),
            (
                "variable",
                -delta_T/battery_capacity/battery_discharge_efficiency,
                dict(
                    name="battery_discharge_power",
                    timestep=self.timesteps_minus_half_hour,
                    scenario=self.scenarios,
                )
            ),
            (
                "variable",
                battery_charge_efficiency * delta_T / battery_capacity,
                dict(
                    name="battery_charge_power",
                    timestep=self.timesteps_minus_half_hour,
                    scenario=self.scenarios,
                )
            ),
            broadcast=["scenario", "timestep"],
        )

        self.optimization_problem.define_constraint(
            (
                "variable",
                1,
                dict(
                    name="soc",
                    timestep=self.timesteps_minus_half_hour[0],
                    scenario=self.scenarios,
                )
            ),
            "==",
            (
                "constant",
                0,
            ),
            broadcast=["scenario"],
        )

        # constraint 1i
        self.optimization_problem.define_constraint(
            (
                "variable",
                1,
                dict(
                    name="soc",
                    timestep=self.timesteps_minus_half_hour,
                    scenario=self.scenarios,
                )
            ),
            ">=",
            (
                "constant",
                0,
            ),
            broadcast=["scenario"],
        )

        self.optimization_problem.define_constraint(
            (
                "variable",
                1,
                dict(
                    name="soc",
                    timestep=self.timesteps_minus_half_hour,
                    scenario=self.scenarios,
                )
            ),
            "<=",
            (
                "constant",
                1,
            ),
            broadcast=["scenario"],
        )

        # constraint 1j)
        self.optimization_problem.define_constraint(
            (
                "variable",
                delta_T*np.ones([1,len(self.timesteps_minus_half_hour)]),
                dict(
                    name="battery_charge_power",
                    timestep=self.timesteps_minus_half_hour,
                    scenario=self.scenarios,
                )
            ),
            (
                "variable",
                delta_T*np.ones([1,len(self.timesteps_minus_half_hour)]),
                dict(
                    name="battery_discharge_power",
                    timestep=self.timesteps_minus_half_hour,
                    scenario=self.scenarios,
                )
            ),
            "<=",
            (
                "constant",
                life_cycles,
            ),
            broadcast=["scenario"],
        )

        mesmo.utils.logger.info('Define objective function')

        market_3_price_recreated = np.repeat(market_3_price, 48)

        self.optimization_problem.define_objective(
            (
                'variable',
                degradation_cost_coefficient*np.ones([1,len(self.timesteps_minus_half_hour)]),
                dict(name='battery_charge_power', timestep=self.timesteps_minus_half_hour)
            ),
            (
                'variable',
                degradation_cost_coefficient*np.ones([1,len(self.timesteps_minus_half_hour)]),
                dict(name='battery_discharge_power', timestep=self.timesteps_minus_half_hour)
            ),
            (
                'variable',
                market_1_price.reshape(1,len(market_1_price))*delta_T,
                dict(name='import_power_market_1', timestep=self.timesteps_minus_half_hour)
            ),
            (
                'variable',
                -market_1_price.reshape(1,len(market_1_price))*delta_T,
                dict(name='export_power_market_1', timestep=self.timesteps_minus_half_hour)
            ),
            (
                'variable',
                market_2_price.reshape(1, len(market_2_price)) * delta_T,
                dict(name='import_power_market_2', timestep=self.timesteps_minus_half_hour)
            ),
            (
                'variable',
                -market_2_price.reshape(1, len(market_2_price)) * delta_T,
                dict(name='export_power_market_2', timestep=self.timesteps_minus_half_hour)
            ),
            (
                'variable',
                market_3_price_recreated.reshape(1, len(market_3_price_recreated)) * delta_T,
                dict(name='import_power_market_3', timestep=self.timesteps_minus_half_hour)
            ),
            (
                'variable',
                -market_3_price_recreated.reshape(1, len(market_3_price_recreated)) * delta_T,
                dict(name='export_power_market_3', timestep=self.timesteps_minus_half_hour)
            ),
        )

        # for evaluation
        self.degradation_cost_vector = degradation_cost_coefficient*np.ones([1,len(self.timesteps_minus_half_hour)])
        self.market_price_1_vector = market_1_price.reshape(1,len(market_1_price))*delta_T
        self.market_price_2_vector = market_2_price.reshape(1, len(market_2_price)) * delta_T
        self.market_price_3_vector = market_3_price_recreated.reshape(1, len(market_3_price_recreated)) * delta_T



        mesmo.utils.logger.info('model defined!')

def main():
    ...


if __name__ == '__main__':
    main()
