# Data

## AirQualityUCI.csv

| | |
|---|---|
| Dataset | Air Quality (UCI Machine Learning Repository, dataset id 360) |
| Creator | De Vito, S. |
| Dataset page | https://archive.ics.uci.edu/dataset/360/air+quality |
| Downloaded from | https://archive.ics.uci.edu/static/public/360/air+quality.zip |
| Retrieved | 2026-09-23 |
| License | Creative Commons Attribution 4.0 International (CC BY 4.0), https://creativecommons.org/licenses/by/4.0/ |
| File committed | `AirQualityUCI.csv` from the zip archive, unmodified (the `.xlsx` copy in the same archive is not included) |
| SHA256 | `13277ae5d8581e80b7be09d47c7d3d06fe9b8e957078f2cf6e859f955e62f996` |
| Size | 785065 bytes |

Check the file with:

```
sha256sum data/AirQualityUCI.csv
```

`.gitattributes` tells Git not to convert the file's line endings, so its CRLF
line endings, and therefore its hash, are preserved on checkout.

### Citation

De Vito, S. (2008). Air Quality [Dataset]. UCI Machine Learning Repository.
https://archive.ics.uci.edu/dataset/360/air+quality

Related paper describing the recording campaign:
S. De Vito, E. Massera, M. Piga, L. Martinotto, G. Di Francia, "On field
calibration of an electronic nose for benzene estimation in an urban pollution
monitoring scenario", Sensors and Actuators B: Chemical 129(2), 2008, 750-757.

### What the file contains

Hourly averages from a multisensor device with five metal-oxide sensors
(`PT08.S1` to `PT08.S5`), recorded alongside a reference analyser station that
measured CO, non-methane hydrocarbons, benzene, NOx and NO2 (the `(GT)`
columns), plus temperature `T`, relative humidity `RH` and absolute humidity
`AH`.

Format details that the loader in `src/aqcal/dataset.py` handles (checked on
this copy of the file):

- `;` separates fields and `,` is the decimal mark;
- the header names 15 columns, but every line ends with `;;`, i.e. two extra
  empty fields;
- line endings are CRLF;
- the header is followed by 9357 data rows (2004-03-10 18:00 to 2005-04-04
  14:00) and then 114 lines that contain only separators;
- missing values are written as `-200`.

Missing values per column in the 9357 data rows, counted after converting
`-200` to missing:

| Column | Missing |
|---|---|
| CO(GT) | 1683 |
| PT08.S1(CO), PT08.S2(NMHC), PT08.S3(NOx), PT08.S4(NO2), PT08.S5(O3) | 366 each |
| NMHC(GT) | 8443 |
| C6H6(GT) | 366 |
| NOx(GT) | 1639 |
| NO2(GT) | 1642 |
| T, RH, AH | 366 each |
