import openap.prop
import openap
from tqdm import tqdm
from functools import cache
from preprocessing.base_preprocessor import BasePreprocessor
from utils.dataset import Dataset

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


class AircraftPerformancePreprocessor(BasePreprocessor):

    @cache
    def props_for_aircraft(self, aircraft_type: str) -> dict:
        try:
            return openap.prop.aircraft(aircraft_type)
        except ValueError:
            return ADDITIONAL_DATA[aircraft_type]

    def process(self, dataset: Dataset) -> Dataset:
        tqdm.pandas()

        dataset.df["mtow"] = dataset.df["aircraft_type"].progress_apply(
            lambda x: self.props_for_aircraft(x)["mtow"]
        )
        dataset.df["mlw"] = dataset.df["aircraft_type"].progress_apply(
            lambda x: self.props_for_aircraft(x)["mlw"]
        )
        return dataset
