import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import mesmo
import csv
import time

from bscs_data_interface import data_bscs, data_ev_swapping_demand_simulation
from bscs_models import bscs_wep_optimization_model, bscs_wep_baseline_simulation_model, \
    bscs_contigency_reserve_optimization_model, bscs_primary_reserve_optimization_model
import numpy as np

def main():
    # Settings
    # select optimisation models from options: wep_model: flag = 0, contingency_reserve: flag = 1, primary reserve = 2
    model_option = 0

    # set run number
    scenario_run_number = 1
    solver_time_save = np.zeros(scenario_run_number)

    # Settings.
    scenario_name = 'bscs_modelling'
    mesmo.data_interface.recreate_database()

    # get time steps for market
    der_model_set = mesmo.der_models.DERModelSet(scenario_name)
    time_step = der_model_set.timesteps

    # Obtain battery swapping station parameter data.
    data_set = data_bscs(os.path.join(os.path.dirname(os.path.normpath(__file__)), 'Dataset'))

    for run_index in range(scenario_run_number):

        # EV swapping demand simulation # randomly generated EV demand #
        data_set_swapping_demand = data_ev_swapping_demand_simulation(time_step.delete(-1))
        # convert data format to df
        imcoming_ev_energy_simulated_df = pd.DataFrame.from_dict(data_set_swapping_demand.data_energy_ev_to_be_swapped_dict, orient='index')

        # simulation model run:
        bscs_simulated = bscs_wep_baseline_simulation_model(scenario_name, data_set, data_set_swapping_demand, time_step)

        # select model
        if model_option == 1:
            bscs_problem = bscs_contigency_reserve_optimization_model(scenario_name, data_set, data_set_swapping_demand,                                                            time_step)
        elif model_option == 0:
            bscs_problem = bscs_wep_optimization_model(scenario_name, data_set, data_set_swapping_demand, time_step)
        elif model_option == 2:
            bscs_problem = bscs_primary_reserve_optimization_model(scenario_name, data_set, data_set_swapping_demand, time_step)

        # record solver time
        t = time.time()

        bscs_problem.optimization_problem.solve()

        solver_time_elapsed = time.time() - t
        solver_time_save[run_index] = solver_time_elapsed

        results = bscs_problem.optimization_problem.get_results()

        # results handling & plotting
        results_path = mesmo.utils.get_results_path(__file__, scenario_name)

        imcoming_ev_energy_simulated_df.to_csv(os.path.join(results_path, 'incoming_ev_energy.csv'))

        battery_slots_energy_res = results["battery_slot_energy"]
        battery_slots_energy_res_df = pd.DataFrame(battery_slots_energy_res)
        battery_slots_energy_res_df.to_csv(os.path.join(results_path, 'battery_slot_energy.csv'))

        # handle results from specific models
        if model_option == 0:
            scenario_definition = ['deterministic']

        elif model_option == 1:
            scenario_definition = ['maximal_reserve_activation', 'no_reserve_activation']

            contingency_reserve_res = results["contingency_reserve"]
            contingency_reserve_res_df = pd.DataFrame(contingency_reserve_res)
            contingency_reserve_res_df.to_csv(os.path.join(results_path, 'contingency_reserve_res.csv'))

            contingency_reserve_binary_res = results["contingency_reserve_binary"]
            contingency_reserve_binary_res_df = pd.DataFrame(contingency_reserve_binary_res)
            contingency_reserve_binary_res_df.to_csv(os.path.join(results_path, 'contingency_reserve_binary_res.csv'))

        elif model_option == 2:
            scenario_definition = ['deterministic']

            primary_reserve_res = results["swapping_station_primary_reserve_offer"]
            primary_reserve_res_df = pd.DataFrame(primary_reserve_res)
            primary_reserve_res_df.to_csv(os.path.join(results_path, 'primary_reserve_res.csv'))

            battery_charge_primary_reserve_res = results["battery_charge_primary_reserve"]
            battery_charge_primary_reserve_res_df = pd.DataFrame(battery_charge_primary_reserve_res)
            battery_charge_primary_reserve_res_df.to_csv(os.path.join(results_path, 'battery_charge_primary_reserve_res.csv'))

            battery_discharge_primary_reserve_res = results["battery_discharge_primary_reserve"]
            battery_discharge_primary_reserve_res_df = pd.DataFrame(battery_discharge_primary_reserve_res)
            battery_discharge_primary_reserve_res_df.to_csv(os.path.join(results_path, 'battery_discharge_primary_reserve_res.csv'))

        # handle common results for all models
        charge_power_res = results["battery_charge_power"]
        charge_power_res_df = pd.DataFrame(charge_power_res)
        charge_power_res_df.to_csv(os.path.join(results_path, 'battery_charge_power.csv'))

        discharge_power_res = results["battery_discharge_power"]
        discharge_power_res_df = pd.DataFrame(discharge_power_res)
        discharge_power_res_df.to_csv(os.path.join(results_path, 'discharge_power.csv'))

        soc_res = results["battery_slot_soc"]
        soc_res_df = pd.DataFrame(soc_res)
        soc_res_df.to_csv(os.path.join(results_path, 'soc.csv'))

        A_matrix_res = results["A_matrix"]
        A_matrix_res_df = pd.DataFrame(A_matrix_res)
        A_matrix_res_df.to_csv(os.path.join(results_path, 'A_matrix.csv'))

        Z_matrix_res = results["Z_matrix"]
        Z_matrix_res_df = pd.DataFrame(Z_matrix_res)
        Z_matrix_res_df.to_csv(os.path.join(results_path, 'Z_matrix.csv'))

        swapping_station_total_energy_demand_res = results["swapping_station_total_energy_demand"]
        swapping_station_total_energy_demand_res_df = pd.DataFrame(swapping_station_total_energy_demand_res)
        swapping_station_total_energy_demand_res_df.to_csv(os.path.join(results_path, 'swapping_station_total_energy_demand.csv'))

        swapping_events_res = results["swapping_events_per_slot"]
        swapping_events_res_df = pd.DataFrame(swapping_events_res)
        swapping_events_res_df.to_csv(
            os.path.join(results_path, 'swapping_events_per_battery_slot.csv'))

        # Plots
        # plots # 0 incoming vehicles number
        fig = px.line(x=time_step.delete(-1),
                        y=data_set_swapping_demand.data_number_ev_to_be_swapped_dict.values(),
                        labels=dict(x="time step", y="incoming vehicle number", variable="Day index"))

        fig.show()

        # plots # 1 swapping events
        fig = px.scatter(x=time_step.delete(-1),
                        y=swapping_events_res_df[scenario_definition[0], 'battery_slot_no_0'].values,
                        labels=dict(x="time step", y="swapping events", variable="Day index"))

        for slot_temp in bscs_problem.battery_slot:
            fig.add_scatter(x=time_step.delete(-1),
                            y=swapping_events_res_df[scenario_definition[0], slot_temp].values,
                            name=slot_temp)

        fig.show()

        # plots # 2 battery slot energy # TODO plot swapping events into figure
        fig = px.line(x=battery_slots_energy_res_df[scenario_definition[0], 'battery_slot_no_0'].index,
                      y=battery_slots_energy_res_df[scenario_definition[0], 'battery_slot_no_0'].values,
                      labels=dict(x="time step", y="battery slot energy - max-res scenario [kWh]", variable="Day index"))

        for slot_temp in bscs_problem.battery_slot:
            fig.add_scatter(x=battery_slots_energy_res_df[scenario_definition[0], slot_temp].index,
                            y=battery_slots_energy_res_df[scenario_definition[0], slot_temp].values,
                            name=slot_temp)

        fig.show()

        # plots # 3 charge results
        fig = px.line(x=charge_power_res_df[scenario_definition[0], 'battery_slot_no_0'].index,
                      y=charge_power_res_df[scenario_definition[0], 'battery_slot_no_0'].values,
                      labels=dict(x="time step", y="charge power [kW]", variable="Day index"))

        for slot_temp in bscs_problem.battery_slot:
            fig.add_scatter(x=charge_power_res_df[scenario_definition[0], slot_temp].index,
                            y=charge_power_res_df[scenario_definition[0], slot_temp].values,
                            name=slot_temp)

        fig.show()

        # plots # 4 discharge results
        fig = px.line(x=discharge_power_res_df[scenario_definition[0], 'battery_slot_no_0'].index,
                      y=discharge_power_res_df[scenario_definition[0], 'battery_slot_no_0'].values,
                      labels=dict(x="time step", y="discharge power [kW]", variable="Day index"))

        for slot_temp in bscs_problem.battery_slot:
            fig.add_scatter(x=discharge_power_res_df[scenario_definition[0], slot_temp].index,
                            y=discharge_power_res_df[scenario_definition[0], slot_temp].values,
                            name=slot_temp)

        fig.show()

        # plots # 5 total energy vs price
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_trace(
            go.Scatter(x=swapping_station_total_energy_demand_res_df[scenario_definition[0]].index,
                       y=swapping_station_total_energy_demand_res_df[scenario_definition[0]].values,
                       name="total BSCS energy consumption"),
            secondary_y=False,
        )

        fig.add_trace(
            go.Scatter(x=swapping_station_total_energy_demand_res_df[scenario_definition[0]].index,
                       y=bscs_problem.wep, name="energy price"),
            secondary_y=True,
        )

        fig.update_yaxes(title_text="battery slot energy [kWh]", secondary_y=False)
        fig.update_yaxes(title_text="energy price SGD/MWh", secondary_y=True)

        fig.show()

        if model_option == 1:
            fig = make_subplots(specs=[[{"secondary_y": True}]])

            fig.add_trace(
                go.Scatter(x=contingency_reserve_res_df[scenario_definition[0]].index,
                           y=contingency_reserve_res_df[scenario_definition[0]].values,
                           name="BSCS contingency reserve offer"),
                secondary_y=False,
            )

            fig.add_trace(
                go.Scatter(x=swapping_station_total_energy_demand_res_df[scenario_definition[0]].index,
                           y=bscs_problem.reserve_price, name="contingency reserve price"),
                secondary_y=True,
            )

            fig.update_yaxes(title_text="battery slot reserve [kWh]", secondary_y=False)
            fig.update_yaxes(title_text="reserve price SGD/MWh", secondary_y=True)

            fig.show()

        if model_option == 2:
            fig = make_subplots(specs=[[{"secondary_y": True}]])

            fig.add_trace(
                go.Scatter(x=primary_reserve_res_df[scenario_definition[0]].index,
                           y=primary_reserve_res_df[scenario_definition[0]].values,
                           name="BSCS primary reserve offer"),
                secondary_y=False,
            )

            fig.add_trace(
                go.Scatter(x=swapping_station_total_energy_demand_res_df[scenario_definition[0]].index,
                           y=bscs_problem.reserve_price, name="primary reserve price"),
                secondary_y=True,
            )

            fig.update_yaxes(title_text="reserve offer quantity [kWh]", secondary_y=False)
            fig.update_yaxes(title_text="reserve price SGD/MWh", secondary_y=True)

            fig.show()

        # generate report in txt files for optimisation and simulation respectively
        # For optimisation:
        # Total energy cost, wep in SGD/MWh, energy demand in kWh
        total_energy_cost = 1/1000*bscs_problem.wep.reshape(1,bscs_problem.wep.size)@swapping_station_total_energy_demand_res.values
        total_energy = swapping_station_total_energy_demand_res.values.sum()

        total_swapping_number = np.where(A_matrix_res.values >= 0.999)[0].size
        total_incoming_vehicle_number = sum(data_set_swapping_demand.data_number_ev_to_be_swapped_dict.values())

        with open(os.path.join(results_path, 'conclusion_optimisation.txt'), 'w') as csvfile:
            writer = csv.writer(csvfile)
            print('Optimisation: Total energy cost (wholesale price only): ', total_energy_cost[0],
                  'SGD for total energy consumption of', [total_energy], 'kWh', file=csvfile)

            print('Optimisation: Swapping demand acceptance rate: ',
                  [100 * total_swapping_number / total_incoming_vehicle_number],
                  '% - {} swapped vehicles - for total number of'.format(total_swapping_number),
                  [total_incoming_vehicle_number], 'vehicles', file=csvfile)

        # For simulation:

        energy_simulated = np.array(list(bscs_simulated.energy_consumption.values()))
        total_energy_cost = 1 / 1000 * bscs_problem.wep.reshape(1,bscs_problem.wep.size) @ energy_simulated
        total_energy = energy_simulated.sum()

        with open(os.path.join(results_path, 'conclusion_simulation.txt'), 'w') as csvfile:
            writer = csv.writer(csvfile)
            print('Simulation: Total energy cost (wholesale price only): ', total_energy_cost[0],
                  'SGD for total energy consumption of', [total_energy], 'kWh', file=csvfile)

            print('Simulation: Swapping demand acceptance rate: ', [100*(total_incoming_vehicle_number-bscs_simulated.rejected_ev_number)/total_incoming_vehicle_number],
                  '% - {} swapped vehicles - for total number of'.format(total_incoming_vehicle_number-bscs_simulated.rejected_ev_number), [total_incoming_vehicle_number], 'vehicles', file=csvfile)

    run_time_result_path = os.path.dirname(results_path)
    with open(os.path.join(run_time_result_path, 'run_time_record.csv'), 'w') as csvfile:
        writer = csv.writer(csvfile)
        print(solver_time_save, file=csvfile)

    print("run ends")


if __name__ == '__main__':
    main()
