import openap.prop
import openap
from tqdm import tqdm
from functools import cache
from preprocessing.base_preprocessor import BasePreprocessor
from utils.dataset import Dataset
import json
from pathlib import Path
import re

ADDITIONAL_DATA = {
    # https://www.airbus.com/en/who-we-are/our-history/commercial-aircraft-history/previous-generation-aircraft/a310
    "A310": {
        "mtow": 164000,
        "mlw": 140000,
    },
    # https://skybrary.aero/aircraft/at76
    # https://doc8643.com/aircraft/AT76
    # https://skyteamvirtual.org/fleet/models/tarom-atr-72-600
    "AT76": {
        "mtow": 23000,
        "mlw": 22350,
    },
    # https://skyteamvirtual.org/fleet/models/delta-air-lines-airbus-a220-100
    "BCS1": {"mtow": 60781, "mlw": 52390},
    # https://skyteamvirtual.org/fleet/models/air-france-airbus-a220-300
    "BCS3": {"mtow": 67585, "mlw": 58740},
    # https://skyteamvirtual.org/fleet/models/delta-private-jets-cessna-citation-excel
    "C56X": {"mtow": 9163, "mlw": 8247},
    # https://skyteamvirtual.org/fleet/models/delta-air-lines-canadair-crj-900
    "CRJ9": {"mtow": 37012, "mlw": 36968},
    # https://www.embraercommercialaviation.com/commercial-jets/e190-e2-commercial-jet/
    "E290": {"mtow": 56400, "mlw": 49050},
}


def parse_specs(specs):
    result = {}

    class_names = [
        "First_Class",
        "First_Class_Suite",
        "Business_Class",
        "Economy_Comfort_Class",
        "Economy_Family_Couch",
        "Premium_Economy_Class",
        "Economy_Class",
        "Total",
    ]
    for name in class_names:
        result[f"Seats {name}"] = 0

    for key, value in specs.items():
        if key == "Cabin Configuration":
            configurations: str = specs[key]
            configurations = configurations.replace(": ", ":")
            configurations = configurations.replace(" ", "_")
            configurations = dict(
                map(
                    lambda x: (x[0], int(x[1])),
                    re.findall(r"(\w+(?:_\w+)*):(\d+)", configurations),
                )
            )
            for name in class_names:
                result[f"Seats {name}"] = configurations.get(name, 0)

            for k in configurations.keys():
                assert k in class_names
        elif key == "Passengers (Cockpit Crew)":
            result["Seats Total"] = int(value.replace("\n", " ").split(" ")[0])
        elif key in [
            "Cargo Capacity",
            "MLW",
            "MTOW",
            "ZFW",
            "Fuel Capacity",
            "Fuel Flow",
        ]:
            # Parse weight values
            result[key] = float(value.split()[0].replace(",", ""))
        elif key == "Range":
            # Parse range
            result[key + "(nm)"] = float(value.split()[0].replace(",", ""))
        elif key == "Service Ceiling":
            # Parse service ceiling
            result[key + "(ft)"] = float(value.split()[0].replace(",", ""))
        elif key == "Cruising Speed":
            # Parse cruising speed
            result[key + "(mach)"] = float(value.split()[1])
        elif key == "Cost Index":
            # Parse cost index
            result[key] = int(value)
        else:
            # Keep other values as strings
            result[key] = value
    return result


def transform_json(input_list):
    output_dict = {}
    for item in input_list:
        icao_code = item["specifications"]["ICAO code"]
        airline = item["specifications"]["Airline"]

        # Create a new dictionary excluding 'Airline' and 'ICAO code' keys
        specifications = {
            key: value
            for key, value in item["specifications"].items()
            if key not in ["Airline", "ICAO code"]
        }

        specifications = parse_specs(specifications)

        # If icao_code is not already in output_dict, initialize it with an empty list
        if icao_code not in output_dict:
            output_dict[icao_code] = {}

        # Append a tuple of airline and specifications to the list for the given icao_code
        output_dict[icao_code][airline] = specifications

    return output_dict


class AircraftPerformancePreprocessor(BasePreprocessor):
    def __init__(self, no_cache=False) -> None:
        super().__init__(no_cache)
        base_dir = Path(__file__).parents[1] / "additional_data" / "aircraft_data"
        if not base_dir.exists():
            raise FileNotFoundError(f"Can not find {base_dir}")
        self.info = json.load(open(base_dir / "aircraft_info.json"))
        manual_info = json.load(open(base_dir / "manual_aircraft_info.json"))

        self.info += manual_info
        self.info = transform_json(self.info)

        self.airline_lut = {
            "a73f82288988b79be490c6322f4c32ed": "Aer Lingus (EIN/EI)",
            "8be5c854fd664bcb97fb543339f74770": "Scandinavian Airlines (SAS/SK)",
            "5d407cb11cc29578cc3e292e743f5393": "Austrian Airlines (AUA/OS)",
            "bdeeef3a675587d530de70a25d7118d2": "Brussels Airlines (BEL/SN)",
            "2d5def0a5a844b343ba1b7cc9cb28fa9": "Swiss (SWR/LX)",
            "3922524069809ac4326134429751e26f": "Jet2",
            "6351ec1b849adacc0cbb3b1313d8d39b": "Turkish Airlines (THY/TK)",
            "5543e4dc327359ffaf5b9c0e6faaf0e1": "American Airlines (AAL/AA)",
            "f5c2e765e074db66052862ab3d1c4529": "TuiFly",  #  Germany",
            "1332254e11e92b4ac6410613b2e86787": "Scandinavian Airlines (SAS/SK)",  # Cityjet
            "f53c55b5cf0cbb3be755bf50df6fa52d": "TuiFly",
            "e36f387a48050121d2415f3935000bdc": "Smartwings (TVS/QS)",
            "8c4e5298059ae6c9ddf6a4ce9a57d1c8": "TuiFly",  # "Tui Airlines",
            "4fea233a1f67230add909d3e8fc8e230": "TuiFly",  # "TuiFly Nordic",
            "5ab5177074e7490ebf8c249ce250759e": "Transavia (TRA/HV)",
            "36b364c9ba9ffb2e3e4803cb4e025745": "Brussels Airlines (BEL/SN)",  # "Air Baltic / Brussels",
            "3a6435cd8884f0dd51b886b3e57267f3": "N/A",
            "415bb6c2faf8f0aa7b4108deeec9869c": "N/A",
            "12838ccf020bc42e0e45c59a5fdf7e82": "N/A",
            "b37a3f3161e6ec4cffbb65e7ebf4ecfe": "N/A",
            "154acc473ac7d5991245125f4ff6b3a6": "N/A",
            "713b84080a5509415d149fe1f7f0add1": "N/A",
            "588c4a7c5b7320c61a6c4227be465964": "N/A",
            "6a681ee572c1e4e981cdab3c55b4b422": "N/A",
            "cc0752e0930c0f501873a342d96c13f0": "N/A",
            "72ba06dd5ae13526df103042ce4c535e": "N/A",
            "310d41975a1e6b9b51ca356414d67daf": "N/A",
            "f502877cab405652cf0dd70c2213e730": "N/A",
            "ecae30f8b0a678b4e97d1f7307642d2b": "N/A",
        }

    @cache
    def props_for_aircraft(self, aircraft_type: str) -> dict:
        airline, type = aircraft_type.split("_")
        airline = self.airline_lut[airline]
        # print("\n".join(sorted(self.info[type].keys())))
        if type not in self.info:
            print(f"Type {type} unknown")
            return {"MTOW": 0}

        airline_options = self.info[type]
        if airline in airline_options:
            return airline_options[airline]
        else:
            print(f"No specific {type} found for {airline}")
            return airline_options[list(airline_options.keys())[0]]

        # from IPython import embed

        # embed()
        # exit()  # TODO: Remove Debug
        # try:
        #     return openap.prop.aircraft(aircraft_type)
        # except ValueError:
        #     return ADDITIONAL_DATA[aircraft_type]

    def process(self, dataset: Dataset) -> Dataset:
        tqdm.pandas()

        # combined column
        dataset.df["airline_aircraft"] = (
            dataset.df["airline"] + "_" + dataset.df["aircraft_type"]
        )

        cols = self.props_for_aircraft(dataset.df["airline_aircraft"].iloc[0]).keys()
        for col in cols:
            dataset.df[col] = dataset.df["airline_aircraft"].progress_apply(
                lambda x: self.props_for_aircraft(x)[col]
            )
        return dataset
