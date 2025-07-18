import argparse
import logging
import os
import sys
from datetime import datetime

from utils.config import DT, OBS_LEN, PRED_LEN, SAMPLE_RATE
from utils.generate_csv_utils import (generate_csv_from_fcd, generate_fcd,
                                      generate_routefile, generate_sumocfg)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="")

    parser.add_argument('--num_seconds', type=int, help='', default=1000)
    parser.add_argument('--create_new_vehicle_prob', type=float, help='', default=0.05) # 0.09
    parser.add_argument('--split', type=str, help='train, val or test', default='train')
    parser.add_argument('--random_seed', type=int, help='', default=7)
    parser.add_argument('--map', type=str, help='map name', default='simple_separate_10m.net.xml') # Maps in sumo/map/

    args = parser.parse_args()

    num_seconds = args.num_seconds
    create_new_vehicle_prob = args.create_new_vehicle_prob
    split = args.split
    random_seed = args.random_seed
    net_xml_file = args.map

    split = f'{split}_{int(num_seconds/1000)}k_{net_xml_file[:-8]}' # The generated csv will be saved in a folder

    now = datetime.now().strftime("%m-%d-%H-%M")
    route_file_name = f'{now}-{create_new_vehicle_prob}-{split}-{random_seed}'
    EDGE_FILTER = 'sumo/sumocfg/filter_edges_simple'
    TRAFFIC_SCALE = 1.0     # regulate the traffic flow
    LENGTH_PER_SCENE = (PRED_LEN + OBS_LEN) // SAMPLE_RATE    # obs + pred (seconds)

    logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s', level=logging.INFO)

    # Generate rou.xml and sumocfg files
    # generate_routefile(rou_xml_filename=route_file_name, num_seconds=num_seconds, create_new_vehicle_prob=create_new_vehicle_prob, random_seed=random_seed)
    generate_routefile(rou_xml_filename=route_file_name, num_seconds=num_seconds, create_new_vehicle_prob=create_new_vehicle_prob, random_seed=random_seed, \
        straight_prob=0.8)
    sumocfg_path = generate_sumocfg(net_xml_file, route_file_name)

    for dir in ['csv', 'fcd']:
        os.makedirs(dir, exist_ok = True)

    # Generate a fcd file
    fcd_file = f'fcd/{route_file_name}.xml'
    logging.info(f'Generating {fcd_file}...')
    generate_fcd(sumocfg_path, fcd_file, EDGE_FILTER, 0, 0, num_seconds, DT, TRAFFIC_SCALE)

    # Generate csv files
    logging.info(f'Generating csv files in csv/{split}...')    
    generate_csv_from_fcd(fcd_file, LENGTH_PER_SCENE, split)
