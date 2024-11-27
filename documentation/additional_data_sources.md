# Data Sources Overview

This document provides an overview of the various data sources we used.

## Aircraft and Airport Data

### AirportsData
- **Description**: Extensive database containing current data for 28,237 airports and landing strips worldwide, including ICAO/IATA codes, names, locations, elevations, coordinates, and timezone information in IANA-compliant format
- **URL**: [airportsdata on PyPI](https://pypi.org/project/airportsdata/)
- **License**: MIT
- **Source**: Fork of [mwgg/Airports](https://github.com/mwgg/Airports), validated against national Aeronautical Information Publications (AIP)

### TimezoneFinder
- **Description**: Python package that enables fast offline timezone lookups based on geographical coordinates
- **URL**: [GitHub - timezonefinder](https://github.com/jannikmi/timezonefinder)
- **License**: MIT
- **Source**: Maintained by jannikmi

### OurAirports - Airports Database
- **Description**: Comprehensive worldwide airports database containing detailed airport information and geographical data
- **URL**: [OurAirports Airports Data](https://davidmegginson.github.io/ourairports-data/airports.csv)
- **License**: Public Domain
- **Source**: OurAirports project

### OurAirports - Runways Database
- **Description**: Detailed global database of airport runway information and specifications
- **URL**: [OurAirports Runways Data](https://davidmegginson.github.io/ourairports-data/runways.csv)
- **License**: Public Domain
- **Source**: OurAirports project

## Weather and Environmental Data

### Iowa Environmental Mesonet (IEM)
- **Description**: Comprehensive collection of historical METAR weather data with API access for programmatic retrieval
- **URL**: [http://mesonet.agron.iastate.edu](http://mesonet.agron.iastate.edu)
- **License**: Public Domain
- **Source**: Iowa State University

## Flight Statistics and Traffic Data

### Eurostat Aviation
- **Description**: Comprehensive European aviation statistics covering airport passenger numbers and traffic data
- **URL**: [Eurostat Aviation API](https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data/avia_tf_apal)
- **License**: Eurostat standard license (free reuse with attribution)
- **Source**: European Commission

### Virtual Airline Databases
- **Description**: Collection of aircraft configuration data including detailed seat layouts from flight simulation communities
- **URLs**: 
  - [One World Virtual](https://oneworldvirtual.org)
  - [Star Alliance Virtual](https://staralliancevirtual.org)
  - [SkyTeam Virtual](https://skyteamvirtual.org)
- **License**: Public Domain
- **Source**: Flight simulation communities

### BTS T-100 Domestic Segment Data
- **Description**: Detailed U.S. air carrier traffic statistics including passenger numbers, freight, capacity, departures, and load factors
- **URL**: [BTS Data](https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FMG)
- **License**: Public Domain
- **Source**: U.S. Bureau of Transportation Statistics

## Reference Data

### ISO Country Codes
- **Description**: Comprehensive dataset of country codes and their regional classifications
- **URL**: [ISO-3166 Countries Dataset](https://raw.githubusercontent.com/lukes/ISO-3166-Countries-with-Regional-Codes/master/all/all.csv)
- **License**: MIT
- **Source**: GitHub repository by lukes

### Fuel Prices Data
- **Description**: Global compilation of petrol and gas prices as of June 2022
- **URL**: [Kaggle Dataset](https://www.kaggle.com/datasets/zusmani/petrolgas-prices-worldwide)
- **License**: CC0 (Creative Commons Zero)
- **Source**: Kaggle

## Additional Resources

### Acropole
- **Description**: Advanced model for aircraft fuel flow prediction with tools for trajectory processing and fuel flow enhancement
- **URL**: [GitHub - Acropole](https://github.com/DGAC/Acropole)
- **License**: AGPL-3.0
- **Source**: DGAC

### Google Destination Insights
- **Description**: Dataset providing travel demand information and search trends for specific routes
- **URL**: [Destination Insights](https://destinationinsights.withgoogle.com/intl/en_ALL/)
- **License**: Unknown
- **Source**: Google

---
Note: Data sources should be properly attributed when used. Verify current licensing terms before use as they may change over time.
