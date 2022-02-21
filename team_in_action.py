from algo_module import Algo_manager
from datetime import datetime, timedelta
from time import perf_counter
from time import sleep
import json
import os
import warnings
from kiteconnect import KiteConnect
warnings.filterwarnings("ignore")

def generate_access_token(broker_secret, request_token) -> str:

        kite = KiteConnect(api_key=broker_secret['kite_api_key'])

        data = kite.generate_session(request_token=request_token, \
            api_secret=broker_secret['kite_api_secret'])
        print("Token Generated")
        del kite 

        return data['access_token']


def execute_algo (**kwargs):

    parent = os.path.dirname(os.getcwd())
    file_path = os.path.join(parent,kwargs['broker_secret_file_name'])

    with open (file_path, "r") as openfile:
        broker_secret = json.load(openfile)

    if kwargs['broker_for_data'] == 'zerodha' or kwargs['broker_for_trade'] == 'zerodha':
        if not kwargs['is_kite_access_token_available']:
            broker_secret['kite_access_token'] = \
                generate_access_token(broker_secret,kwargs['kite_request_token'])

        with open (file_path, "w") as outfile:
            json.dump(broker_secret,outfile)

    # logs_folder_path = kwargs['log_folder_name']
    logs_folder_path = os.path.join(parent,kwargs['log_folder_name'])

    if kwargs['broker_for_data'].upper() == 'SIM':
        current_datetime = kwargs['day_start_datetime']
    else: 
        current_datetime = datetime.now()

    module_initiation_time = datetime.now()
    algo_manager = Algo_manager(
        broker_for_trade=kwargs['broker_for_trade'],
        broker_for_data=kwargs['broker_for_data'],
        per_trade_fee = kwargs['per_trade_fee'],
        underlying_name=kwargs['underlying_name'],
        kotak_consumer_key=broker_secret['kotak_consumer_key'],
        kotak_access_token=broker_secret['kotak_access_token'],
        kotak_consumer_secret=broker_secret['kotak_consumer_secret'],
        kotak_user_id=broker_secret['kotak_user_id'],
        kotak_access_code=broker_secret['kotak_access_code'],
        kotak_user_password=broker_secret['kotak_user_password'],
        kite_api_key=broker_secret['kite_api_key'],
        kite_access_token=broker_secret['kite_access_token'],
        log_folder=logs_folder_path,
        current_datetime = current_datetime,
        broker_connection_loss = kwargs['broker_connection_loss'],
        exchange_connection_loss = kwargs['exchange_connection_loss'],
        entry_datetime=kwargs['entry_datetime'],
        exit_datetime=kwargs['exit_datetime'],
        quantity_per_lot = 50,
        options_step_size = 50,
        big_jump = timedelta(minutes=10),
        small_jump = timedelta(minutes=1),
        is_jumping = kwargs['is_jumping'],
        ltp_to_position_distance = .03,
        underlying_max_movement = .05,
        lots_traded = kwargs['lots_traded'],
        total_loss_limit_per_lot = -10_000,
        trailing_loss_limit_per_lot = -500,
        trailing_loss_trigger_point_per_lot = 5_000,
        historical_data_folder_name = kwargs['historical_data_folder_name'],
        fno_folder_name = kwargs['fno_folder_name'],
        equity_folder_name = kwargs['equity_folder_name']
        )
    print(f'Module Initiation took: {datetime.now()-module_initiation_time}')
    count = 0
    execution_start_time = datetime.now()

    if kwargs['broker_for_data'].upper() == 'SIM':
        current_datetime = kwargs['day_start_datetime']
    else: 
        current_datetime = datetime.now()

    while current_datetime <= kwargs['switch_off_time']:
        if current_datetime.time() > datetime(2021,5,17,15,30).time():
            current_datetime = current_datetime + timedelta (hours=17,minutes=45)
        initiation_time = perf_counter()
        jump_size = algo_manager.action(current_datetime=current_datetime,
            initiation_time = initiation_time)

        print(current_datetime.strftime('%Y-%b-%d>%I:%M:%S %p, %a                '),
            algo_manager.events_and_actions.display_string(),end='\r')
        
        count += 1
        
        if kwargs['broker_for_data'].upper() == 'SIM':
            slippage = perf_counter() - initiation_time
            if jump_size is None:
                jump_size = timedelta(seconds=0)
            current_datetime = current_datetime \
                + timedelta(\
                    seconds=(slippage+\
                    kwargs['pause_between_iterations']))\
                + jump_size
        else: 
            sleep(kwargs['pause_between_iterations'])
            current_datetime = datetime.now()
    
    time_elapsed = datetime.now() - execution_start_time
    iterations_per_minute = round(count/(time_elapsed.seconds/60),0)
    print(f"Total Time: {time_elapsed}, Iterations: {count}, Per Minute: {iterations_per_minute}")
    
    if kwargs['broker_for_data'].upper() == 'SIM':
        time_elapsed = current_datetime - kwargs['day_start_datetime']
        iterations_per_minute = round(count/(time_elapsed.seconds/60),0)
        print(f"Simulated Time: {time_elapsed}, Iterations: {count}, Per Simulated Minute: {iterations_per_minute}")

    print(f"Days: Profit : {algo_manager.data_guy.strategy_pnl} + {algo_manager.data_guy.brokerage_pnl}")
    pass


if __name__ == '__main__':

    #For Simulation
    day_start_datetime = datetime(2021,5,21,9,15)
    entry_datetime = datetime(2021,5,21,9,30)
    exit_datetime = datetime(2021,5,27,15,15)
    switch_off_time =    datetime(2021,5,27,15,27)

    # For Live Paper trade
    # day_start_datetime = None
    # trading_start_time = datetime(2020,1,1,12,0).time()
    # trading_close_time = datetime(2020,1,1,15,7).time()
    # switch_off_time =    datetime(2020,1,1,15,10).time()


    is_kite_access_token_available = False
    kite_request_token='9ssq7Xyih3uC16rtS23BdiO6wZr5cj5c'

    broker_secret_file_name = 'broker_secret.json'

    log_folder_name = 'logs'

    per_trade_fee = -20
    lots_traded = 1
    underlying_name = 'NIFTY'

    broker_for_trade = 'paper'
    broker_for_data = 'sim'

    pause_between_iterations = .1 

    broker_connection_loss = None
    exchange_connection_loss = None


    # broker_connection_loss = [{'start_datetime':datetime(2021,5,17,9,16),
    #                             'end_datetime':datetime(2021,5,17,9,18)},
    #                         {'start_datetime':datetime(2021,5,17,9,21),
    #                         'end_datetime':datetime(2021,5,17,9,32)},
    #                         {'start_datetime':datetime(2021,5,17,10,22),
    #                         'end_datetime':datetime(2021,5,17,10,23)},
    #                         {'start_datetime':datetime(2021,5,17,11,32),
    #                         'end_datetime':datetime(2021,5,17,12,50)}]

    

    historical_data_folder_name = 'historical data'
    fno_folder_name = 'FNO'
    equity_folder_name = 'Equity'

    execute_algo (day_start_datetime=day_start_datetime,
        entry_datetime=entry_datetime,
        exit_datetime = exit_datetime,
        switch_off_time = switch_off_time,
        is_kite_access_token_available = is_kite_access_token_available,
        kite_request_token = kite_request_token,
        broker_secret_file_name = broker_secret_file_name,
        log_folder_name = log_folder_name,
        per_trade_fee = per_trade_fee,
        lots_traded = lots_traded,
        broker_for_trade = broker_for_trade,
        broker_for_data = broker_for_data,
        underlying_name = underlying_name,
        pause_between_iterations = pause_between_iterations,
        historical_data_folder_name = historical_data_folder_name,
        fno_folder_name = fno_folder_name,
        equity_folder_name = equity_folder_name,
        broker_connection_loss = broker_connection_loss,
        exchange_connection_loss = exchange_connection_loss,
        is_jumping = True)