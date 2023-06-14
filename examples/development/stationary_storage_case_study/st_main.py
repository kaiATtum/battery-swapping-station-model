import os
import pandas as pd
import plotly.graph_objects as go
import mesmo
import numpy as np

from st_data_interface import data_stationary_storage
from st_model import stationary_storage_wep_optimization_model


def main():

    # Settings
    # Base settings.
    scenario_name = 'bscs_modelling'
    mesmo.data_interface.recreate_database()

    # Obtain data.
    data_set = data_stationary_storage(os.path.join(os.path.dirname(os.path.normpath(__file__)), 'Dataset'))
    problem = stationary_storage_wep_optimization_model(scenario_name, data_set)

    problem.optimization_problem.solve()

    results = problem.optimization_problem.get_results()

    results_charge_power = results["battery_charge_power"]
    results_discharge_power = results["battery_discharge_power"]
    results_import_m1 = results["import_power_market_1"]
    results_export_m1 = results["export_power_market_1"]
    results_import_m2 = results["import_power_market_2"]
    results_export_m2 = results["export_power_market_2"]
    results_import_m3 = results["import_power_market_3"]
    results_export_m3 = results["export_power_market_3"]

    results_path = mesmo.utils.get_results_path(__file__, scenario_name)

    # combine results:
    df = pd.DataFrame(
        {
            'time_step': results_charge_power.index,
            'battery_charge_power [MW]': results_charge_power.values.reshape(results_charge_power.values.shape[0]),
            'battery_discharge_power [MW]': results_discharge_power.values.reshape(results_charge_power.values.shape[0]),
            'import_power_market_1 [MW]': results_import_m1.values.reshape(results_charge_power.values.shape[0]),
            'export_power_market_1 [MW]': results_export_m1.values.reshape(results_charge_power.values.shape[0]),
            'import_power_market_2 [MW]': results_import_m2.values.reshape(results_charge_power.values.shape[0]),
            'export_power_market_2 [MW]': results_export_m2.values.reshape(results_charge_power.values.shape[0]),
            'import_power_market_3 [MW]': results_import_m3.values.reshape(results_charge_power.values.shape[0]),
            'export_power_market_3 [MW]': results_export_m3.values.reshape(results_charge_power.values.shape[0]),
        }
    )

    # df.set_index('time_step')
    degradation_cost = (problem.degradation_cost_vector@results_discharge_power.values.reshape([results_discharge_power.values.shape[0],1]))[0] \
                       + (problem.degradation_cost_vector@results_charge_power.values.reshape([results_charge_power.values.shape[0],1]))[0]

    market_1_revenue = -(problem.market_price_1_vector@results_import_m1.values.reshape([results_import_m1.values.shape[0],1]))[0] + \
                       (problem.market_price_1_vector@results_export_m1.values.reshape([results_export_m1.values.shape[0],1]))[0]

    market_2_revenue = -(problem.market_price_2_vector @ results_import_m2.values.reshape([results_import_m2.values.shape[0], 1]))[0] + \
                       (problem.market_price_2_vector @ results_export_m2.values.reshape([results_export_m2.values.shape[0], 1]))[0]

    market_3_revenue = -(problem.market_price_3_vector @ results_import_m3.values.reshape([results_import_m3.values.shape[0], 1]))[0] + \
                       (problem.market_price_3_vector @ results_export_m3.values.reshape([results_export_m3.values.shape[0], 1]))[0]

    fixed_cost_per_year = data_set.battery_data.query('parameter == "Fixed Operational Costs"')["Values"].values[0]

    df2 = pd.DataFrame(
        {
            'total profits [GBP]': [(market_1_revenue + market_2_revenue + market_3_revenue)[0]-degradation_cost[0]-fixed_cost_per_year*3],
            'total market revenue [GBP]': market_1_revenue + market_2_revenue + market_3_revenue,
            'degradation cost [GBP]': degradation_cost,
            'fixed cost [GBP]': [fixed_cost_per_year*3],
            'objective function value': [problem.optimization_problem.objective],
        }
    )

    # save results to excel file:
    with pd.ExcelWriter(os.path.join(results_path, 'results.xlsx') ) as writer:
        df.to_excel(writer, sheet_name='variable_results')
        df2.to_excel(writer, sheet_name='total profits')

    # some plots:
    fig = go.Figure()

    number_of_period = 365*3*48  # 2930 # #48*365
    fig.add_scatter(x=data_set.time_stamp_half_hour[0:number_of_period],  # hour
                    y=results_charge_power.values.reshape(results_charge_power.values.shape[0])[0:number_of_period],
                    name='charge power +',  # kW
                    ).update_traces(mode="lines")

    fig.add_scatter(x=data_set.time_stamp_half_hour[0:number_of_period],  # hour
                    y=-results_discharge_power.values.reshape(results_discharge_power.values.shape[0])[0:number_of_period],
                    name='discharge power -',  # kW
                    ).update_traces(mode="lines")

    fig.update_layout(
        title="charge/discharge results",
        title_x=0.5,
        xaxis_title="time step [h]",
        yaxis_title="power [MW] (charge[+], discharge[-])",
    )

    fig.show()

    fig = go.Figure()

    fig.add_scatter(x=data_set.time_stamp_half_hour[0:number_of_period],  # hour
                    y=results_import_m1.values.reshape(results_import_m1.values.shape[0])[0:number_of_period],
                    name='m1-import',  # kW
                    ).update_traces(mode="lines")

    fig.add_scatter(x=data_set.time_stamp_half_hour[0:number_of_period],  # hour
                    y=-results_export_m1.values.reshape(results_export_m1.values.shape[0])[0:number_of_period],
                    name='m1-export',  # kW
                    ).update_traces(mode="lines")

    fig.add_scatter(x=data_set.time_stamp_half_hour[0:number_of_period],  # hour
                    y=results_import_m2.values.reshape(results_import_m1.values.shape[0])[0:number_of_period],
                    name='m2-import',  # kW
                    ).update_traces(mode="lines")

    fig.add_scatter(x=data_set.time_stamp_half_hour[0:number_of_period],  # hour
                    y=-results_export_m2.values.reshape(results_export_m1.values.shape[0])[0:number_of_period],
                    name='m2-export',  # kW
                    ).update_traces(mode="lines")

    fig.add_scatter(x=data_set.time_stamp_half_hour[0:number_of_period],  # hour
                    y=results_import_m3.values.reshape(results_import_m1.values.shape[0])[0:number_of_period],
                    name='m3-import',  # kW
                    ).update_traces(mode="lines")

    fig.add_scatter(x=data_set.time_stamp_half_hour[0:number_of_period],  # hour
                    y=-results_export_m3.values.reshape(results_export_m1.values.shape[0])[0:number_of_period],
                    name='m3-export',  # kW
                    ).update_traces(mode="lines")

    fig.update_layout(
        title="import-export results",
        title_x=0.5,
        xaxis_title="time steps",
        yaxis_title="power [MW] (import[+], export[-])",
    )

    fig.show()

    mesmo.utils.logger.info('run end.')


if __name__ == '__main__':
    main()
