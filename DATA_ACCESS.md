# Data access

## China: CSMAR China County Statistical Database (not redistributed)

The Chinese county panel (agricultural gross output value, population,
cultivated area, 2000-2023) is built from the China County Statistical
Database of CSMAR (China Stock Market & Accounting Research,
https://data.csmar.com), available through an institutional subscription. Its
licence does not permit redistribution of the county values or of panels
derived from them, so this archive contains none: only aggregates (counts per
province and year, window-level estimates, bootstrap and per-draw summary
statistics) are shipped. Researchers with a subscription can obtain the
database from CSMAR directly; the file, sheet and column names that the
construction code reads are in `src/config.py`, and `make china-panel`
builds the panel from the Excel files placed under `data/`.

## China: list of the 832 national poverty counties (not shipped)

`make china-panel` also reads the list of the 832 national-level poverty
counties with the year each county left the list (file and sheet names in
`src/config.py`, `POVERTY_FILE`; columns: county code, name, region,
province, prefecture, exit year). The list was announced by the State
Council Leading Group Office of Poverty Alleviation and Development (2014),
and the exit years (2016-2020) by the provincial governments; the authors
used a compiled spreadsheet of these public announcements. It is used only
for the poverty indicator columns of the panel (`src/data_merger.add_poverty_status`),
which no table, number or figure of the paper reads. The compiled file is
not redistributed here; it is available from the authors on request.

## United States: BEA and OMB (public domain)

County population and personal income: BEA Regional Economic Accounts table
CAINC1 (https://apps.bea.gov/regional/zip/CAINC1.zip). County GDP: BEA table
CAGDP1 (https://apps.bea.gov/regional/zip/CAGDP1.zip). Metropolitan
delineations: OMB delineation files published by the Census Bureau
(https://www.census.gov/geographies/reference-files/time-series/demo/metro-micro/delineation-files.html
and the historical delineation files). `make us` downloads every file and
checks it; a CAINC1.zip or CAGDP1.zip already placed under
`data/us/raw/bea/` is unpacked instead of downloaded. BEA replaces these
archives with each annual release. The release used is the one with data
through 2024 whose archive members are dated 14 January 2026 (CAINC1,
1969-2024) and 29 January 2026 (CAGDP1, 2001-2024); the hashes below identify
it. Once BEA publishes a later release, the files of the release used can be
requested from BEA's Regional Economic Accounts (https://www.bea.gov/data/by-place-county-metro-local)
or from the authors.

| File (under `data/us/`) | Source | sha256 |
|---|---|---|
| `raw/bea/x/CAGDP1__ALL_AREAS_2001_2024.csv` | https://apps.bea.gov/regional/zip/CAGDP1.zip (member of the zip) | `444b10c643441d6631fd0e3542625e0e30900ef626e89ccea43a907b28750f14` |
| `raw/bea/x/CAINC1__ALL_AREAS_1969_2024.csv` | https://apps.bea.gov/regional/zip/CAINC1.zip (member of the zip) | `98c5ef2a2d3f72e482eac5bca0544ce3114375f8908226872b4f429d0e0f2028` |
| `raw/omb/1993/93mfips.txt` | https://www2.census.gov/programs-surveys/metro-micro/geographies/reference-files/1993/historical-delineation-files/93mfips.txt | `55a3497533d6761228f74c579abf9ca8bde3aaa47bc4a265b702486c859115c7` |
| `raw/omb/1993/93nfips.txt` | https://www2.census.gov/programs-surveys/metro-micro/geographies/reference-files/1993/historical-delineation-files/93nfips.txt | `b28962a6f9533228e71a2fc7d2bff32acd55c180b627508a8b3df527e03021f1` |
| `raw/omb/1999/99mfips.txt` | https://www2.census.gov/programs-surveys/metro-micro/geographies/reference-files/1999/historical-delineation-files/99mfips.txt | `dda547e398819c4040fb2b2d07256eaee76e7504bb236b92926e12b1d28f933c` |
| `raw/omb/1999/99nfips.txt` | https://www2.census.gov/programs-surveys/metro-micro/geographies/reference-files/1999/historical-delineation-files/99nfips.txt | `40652473b359474e3bc427dd6d27997c9b0c080fe62130e6d0cc0ffc106e69c0` |
| `raw/omb/2003/030606omb-cbsa-csa.xls` | https://www2.census.gov/programs-surveys/metro-micro/geographies/reference-files/2003/historical-delineation-files/030606omb-cbsa-csa.xls | `2d8615b4e6a158d767ae46dc2783a5226626ff33f8f3360f9bc524fba01346ef` |
| `raw/omb/200312/0312cbsas-csas.xls` | https://www2.census.gov/programs-surveys/metro-micro/geographies/reference-files/2003/historical-delineation-files/0312cbsas-csas.xls | `c69bf6f5a56815df5221b7f6f86b9c2edd972026db58472ea88dac77ecf1226b` |
| `raw/omb/2004/list3.xls` | https://www2.census.gov/programs-surveys/metro-micro/geographies/reference-files/2004/historical-delineation-files/list3.xls | `9623d5526599c71e4a8ed827cf6393a4575aa16ec7a4053d21201dea5d881461` |
| `raw/omb/2005/list3.xls` | https://www2.census.gov/programs-surveys/metro-micro/geographies/reference-files/2005/historical-delineation-files/list3.xls | `6f539c9b2e0e81410c9d5fe250c95271f8ff29f6f56cc9ec90675300406597a1` |
| `raw/omb/2006/list3.xls` | https://www2.census.gov/programs-surveys/metro-micro/geographies/reference-files/2006/historical-delineation-files/list3.xls | `7e9186f36a4c853a29c4675a2477a15aa7238b09b115476973dac541247425c8` |
| `raw/omb/2007/list3.xls` | https://www2.census.gov/programs-surveys/metro-micro/geographies/reference-files/2007/historical-delineation-files/list3.xls | `873578952eabdda97db3dd0b0ce7f036792833246cb93ccd28f8fe04df6beee6` |
| `raw/omb/2008/list3.xls` | https://www2.census.gov/programs-surveys/metro-micro/geographies/reference-files/2008/historical-delineation-files/list3.xls | `e51e4ceba0cdba0c9fca519343d40141bee4c94661710da96b43b3a16e3a1a4c` |
| `raw/omb/2009/list3.xls` | https://www2.census.gov/programs-surveys/metro-micro/geographies/reference-files/2009/historical-delineation-files/list3.xls | `261ce95a1882d9cb020f5129cd12006717eaf3a29b3b92e4152538b9b0041b90` |
| `raw/omb/2013/list1.xls` | https://www2.census.gov/programs-surveys/metro-micro/geographies/reference-files/2013/delineation-files/list1.xls | `1bb1d7ce747bcd72fc212bda806c28bb4b17d4cec72d1388770f2b0a6d06f5c0` |
| `raw/omb/2015/list1.xls` | https://www2.census.gov/programs-surveys/metro-micro/geographies/reference-files/2015/delineation-files/list1.xls | `3f04531db0b1963c8681ad905bedd2efe2aa01084b9d90cb601f68a0b6d43323` |
| `raw/omb/2017/list1.xls` | https://www2.census.gov/programs-surveys/metro-micro/geographies/reference-files/2017/delineation-files/list1.xls | `0b71f1ca2a59eec6ad393193b16037ca3e80a03a1a5ef1ec6b25292698568eff` |
| `raw/omb/201804/list1.xls` | https://www2.census.gov/programs-surveys/metro-micro/geographies/reference-files/2018/delineation-files/list1.xls | `09521676a26f294274705ddde7c5732e00d4ed1a910a0ad2d4f4b7ee65d2c857` |
| `raw/omb/201809/list1_Sep_2018.xls` | https://www2.census.gov/programs-surveys/metro-micro/geographies/reference-files/2018/delineation-files/list1_Sep_2018.xls | `22ba2e944ce1aa88b0bf0d09530d08f26c17a0bba7ae823290e7c4816316418c` |
| `raw/omb/2020/list1_2020.xls` | https://www2.census.gov/programs-surveys/metro-micro/geographies/reference-files/2020/delineation-files/list1_2020.xls | `95b389487efcd818d71ba2a992e1edd43af93228e70c2a9634fd152acb345409` |
| `raw/omb/2023/list1_2023.xlsx` | https://www2.census.gov/programs-surveys/metro-micro/geographies/reference-files/2023/delineation-files/list1_2023.xlsx | `952c4b1e78acbb54e6ec9412434b7602fedacbf021736351a63c181bdb753629` |

## Brazil: IBGE (public statistics, used with attribution)

Municipal population (SIDRA tables 202, 793, 4709 and 6579 and the FTP
population estimates), the territorial division (DTB) and genealogy of
municipalities, and the PIB dos Municipios, all from the Instituto Brasileiro
de Geografia e Estatistica (https://www.ibge.gov.br). IBGE statistical tables
may be used and reproduced with citation of the source. `make brazil`
downloads every file and checks it.

| File (under `data/brazil/`) | Content | URL | sha256 |
|---|---|---|---|
| `raw/sidra/t202_p1991.json` | municipal population 1991 (census) | https://apisidra.ibge.gov.br/values/t/202/n6/all/v/93/p/1991 | `1e9f191c64835a7e50a7d54941a8895c912fc80d0737e1cceb4b752a43851988` |
| `raw/ftp_estimativas/estimativa_populacao_1992_xls.zip` | municipal population 1992 (estimate_ftp) | https://ftp.ibge.gov.br/Estimativas_de_Populacao/Estimativas_1992/estimativa_populacao_1992_xls.zip | `c460e4aedfa2f59402e0a4c791987de27093055d253dcee96014e844e784da05` |
| `raw/ftp_estimativas/estimativa_populacao_1993_xls.zip` | municipal population 1993 (estimate_ftp) | https://ftp.ibge.gov.br/Estimativas_de_Populacao/Estimativas_1993/estimativa_populacao_1993_xls.zip | `51b53c35a974099eaa2511c56efe3b1ba738977ce8666eba77b4d6270c2e9d38` |
| `raw/ftp_estimativas/estimativa_populacao_1994_xls.zip` | municipal population 1994 (estimate_ftp) | https://ftp.ibge.gov.br/Estimativas_de_Populacao/Estimativas_1994/estimativa_populacao_1994_xls.zip | `220716eb4eaf39ae11e93616cb20464685254218dfe03e87efad8660d063f7b5` |
| `raw/ftp_estimativas/estimativa_populacao_1995_xls.zip` | municipal population 1995 (estimate_ftp) | https://ftp.ibge.gov.br/Estimativas_de_Populacao/Estimativas_1995/estimativa_populacao_1995_xls.zip | `60eb7ba1cb33f6cc1a104f6e16b08d6c59b283b7ccc9fbdbf9ed7173b83081d8` |
| `raw/ftp_estimativas/estimativa_populacao_1997_xls.zip` | municipal population 1997 (estimate_ftp) | https://ftp.ibge.gov.br/Estimativas_de_Populacao/Estimativas_1997/estimativa_populacao_1997_xls.zip | `c8c29cac5d5b5def3cdb49d0775edfc29c844bbb7ce20b4778fd387b583de2c7` |
| `raw/ftp_estimativas/estimativa_populacao_1998_xls.zip` | municipal population 1998 (estimate_ftp) | https://ftp.ibge.gov.br/Estimativas_de_Populacao/Estimativas_1998/estimativa_populacao_1998_xls.zip | `2e60e1d9f355957468b0fb0b6030aa2224e6fbb5a096252a0d38beaf0c40269b` |
| `raw/ftp_estimativas/estimativa_populacao_1999_xls.zip` | municipal population 1999 (estimate_ftp) | https://ftp.ibge.gov.br/Estimativas_de_Populacao/Estimativas_1999/estimativa_populacao_1999_xls.zip | `47040982a5b048a645789a116b0586b702c41b1fb418dae824801982ab6f6c03` |
| `raw/sidra/t202_p2000.json` | municipal population 2000 (census) | https://apisidra.ibge.gov.br/values/t/202/n6/all/v/93/p/2000 | `008aa7a74cda9f2d647a116f53ea3ed3fb3b075e5046366e31e641d816ad1c88` |
| `raw/sidra/t6579_p2001.json` | municipal population 2001 (estimate_sidra6579) | https://apisidra.ibge.gov.br/values/t/6579/n6/all/v/9324/p/2001 | `fc2c524275497a985f6d8ab567525b86990bbb71b957c8bfe42a4e2341f0ec20` |
| `raw/sidra/t6579_p2002.json` | municipal population 2002 (estimate_sidra6579) | https://apisidra.ibge.gov.br/values/t/6579/n6/all/v/9324/p/2002 | `07f45fa3ee3dddcff2ff5e907dae7cd67279042e18584c285e1488f4b8d4a6cd` |
| `raw/sidra/t6579_p2003.json` | municipal population 2003 (estimate_sidra6579) | https://apisidra.ibge.gov.br/values/t/6579/n6/all/v/9324/p/2003 | `b27785e0bcee945425fa2c858a0b7f0cd36337f8d1b1ed270f52450c4b5d99b5` |
| `raw/sidra/t6579_p2004.json` | municipal population 2004 (estimate_sidra6579) | https://apisidra.ibge.gov.br/values/t/6579/n6/all/v/9324/p/2004 | `eac542405fdb603d15d9846f54ed28aa2e857827ce5c51b38a4a81ebcc056b6f` |
| `raw/sidra/t6579_p2005.json` | municipal population 2005 (estimate_sidra6579) | https://apisidra.ibge.gov.br/values/t/6579/n6/all/v/9324/p/2005 | `fd9e1332353c24d06d1ed44d76341dca45284a3a05d1497565ab737266e3a542` |
| `raw/sidra/t6579_p2006.json` | municipal population 2006 (estimate_sidra6579) | https://apisidra.ibge.gov.br/values/t/6579/n6/all/v/9324/p/2006 | `e1e5c1a9b06792f0ef72fcabb0244d55a5eb470a106a60d6586207ff11119b43` |
| `raw/sidra/t793_p2007.json` | municipal population 2007 (contagem) | https://apisidra.ibge.gov.br/values/t/793/n6/all/v/93/p/2007 | `d6fa3ce01497e3f1d15a2c9e524f37386ff11ae588b610d0edad5536c38d9866` |
| `raw/sidra/t6579_p2008.json` | municipal population 2008 (estimate_sidra6579) | https://apisidra.ibge.gov.br/values/t/6579/n6/all/v/9324/p/2008 | `0d8ee0b506beaa90ff8d6ad72b37ff763b58d03c45f44303344ff3b63d053b49` |
| `raw/sidra/t6579_p2009.json` | municipal population 2009 (estimate_sidra6579) | https://apisidra.ibge.gov.br/values/t/6579/n6/all/v/9324/p/2009 | `fd6e3425cb9adbf50e24922c1da4141a6776970fd9f4ee9df1cb981a95871322` |
| `raw/sidra/t202_p2010.json` | municipal population 2010 (census) | https://apisidra.ibge.gov.br/values/t/202/n6/all/v/93/p/2010 | `c6e2de8dae6f966f137cbfcd67b54f4da47b227ffd62bfad74ceb5a6d8087936` |
| `raw/sidra/t6579_p2011.json` | municipal population 2011 (estimate_sidra6579) | https://apisidra.ibge.gov.br/values/t/6579/n6/all/v/9324/p/2011 | `8c1212b823c915e27caa8e5007645e580682201d2acfe8c34ea5529015fcbc03` |
| `raw/sidra/t6579_p2012.json` | municipal population 2012 (estimate_sidra6579) | https://apisidra.ibge.gov.br/values/t/6579/n6/all/v/9324/p/2012 | `b5023e0baa51da728486d0a59daecf32c9e0f54ac3c0295b8d95a49f475e728c` |
| `raw/sidra/t6579_p2013.json` | municipal population 2013 (estimate_sidra6579) | https://apisidra.ibge.gov.br/values/t/6579/n6/all/v/9324/p/2013 | `e75232444432aeedad651f6f5f926badf6e079bf2708fe6285e7e3844915ead8` |
| `raw/sidra/t6579_p2014.json` | municipal population 2014 (estimate_sidra6579) | https://apisidra.ibge.gov.br/values/t/6579/n6/all/v/9324/p/2014 | `c6f947001bfb0a30699799db27e6758a4afe2e0f72fe4cf35bc604624111e28b` |
| `raw/sidra/t6579_p2015.json` | municipal population 2015 (estimate_sidra6579) | https://apisidra.ibge.gov.br/values/t/6579/n6/all/v/9324/p/2015 | `de8827be20796ed33e56329507b938a03725a742f3339209957a336eb3f64f55` |
| `raw/sidra/t6579_p2016.json` | municipal population 2016 (estimate_sidra6579) | https://apisidra.ibge.gov.br/values/t/6579/n6/all/v/9324/p/2016 | `fa484dc099fef2c3105db99002552d83d92ea8f18104444498b8d037429f3f3a` |
| `raw/sidra/t6579_p2017.json` | municipal population 2017 (estimate_sidra6579) | https://apisidra.ibge.gov.br/values/t/6579/n6/all/v/9324/p/2017 | `0afa9c65290fc18ab57dfdf723f8abe8cf581c8111b29b80e3fc9510bbe0044d` |
| `raw/sidra/t6579_p2018.json` | municipal population 2018 (estimate_sidra6579) | https://apisidra.ibge.gov.br/values/t/6579/n6/all/v/9324/p/2018 | `89a4599c4a4591050c605d5825a5a90f7397d4dcf8ff1d87384a3cd7572b5ffb` |
| `raw/sidra/t6579_p2019.json` | municipal population 2019 (estimate_sidra6579) | https://apisidra.ibge.gov.br/values/t/6579/n6/all/v/9324/p/2019 | `313a958cba93409fbff6ff050ba8af7746314dc2219281881fb104ac00ca1ea6` |
| `raw/sidra/t6579_p2020.json` | municipal population 2020 (estimate_sidra6579) | https://apisidra.ibge.gov.br/values/t/6579/n6/all/v/9324/p/2020 | `c1155cfe1217a6b2f6d7fca8d1b25974ffe63fc1434aa0a455de5ae652931b26` |
| `raw/sidra/t6579_p2021.json` | municipal population 2021 (estimate_sidra6579) | https://apisidra.ibge.gov.br/values/t/6579/n6/all/v/9324/p/2021 | `38cd175e7f345de65baec455672bf7204115e2bc9f863f0740628b1e813900d7` |
| `raw/sidra/t4709_p2022.json` | municipal population 2022 (census) | https://apisidra.ibge.gov.br/values/t/4709/n6/all/v/93/p/2022 | `ec3c169ff4e6f59641c3ec681aabe2b96b65edf2273e2f319eaa37d565413558` |
| `raw/sidra/t6579_p2024.json` | municipal population 2024 (estimate_sidra6579) | https://apisidra.ibge.gov.br/values/t/6579/n6/all/v/9324/p/2024 | `b0dd3ad00824c6718c8298b690243d0f7fe6a6f48eb140af631e7412065b4c01` |
| `raw/sidra/t6579_p2025.json` | municipal population 2025 (estimate_sidra6579) | https://apisidra.ibge.gov.br/values/t/6579/n6/all/v/9324/p/2025 | `3a2be712c79be2c8c50bd16527cd5cfcf14a58b0ab4c80bef379dc0f7c7e262d` |
| `raw/sidra/t6579_p2026.json` | municipal population 2026 (estimate_sidra6579) | https://apisidra.ibge.gov.br/values/t/6579/n6/all/v/9324/p/2026 | `2af3c54c40389e6aa25a347ae4ac023e9b1c0b5d7da7eccf76ccd85fd6716a73` |
| `raw/ftp_estimativas/estimativa_2000_UF_Municipio.zip` | territorial division / genealogy | https://ftp.ibge.gov.br/Estimativas_de_Populacao/Estimativas_2000/UF_Municipio.zip | `96a4ecc2d957a6ff469e082455d85bcaa0305093e7f7cf5abe430c51d98f6194` |
| `raw/dtb/dtb_1994_brasil.zip` | territorial division / genealogy | https://geoftp.ibge.gov.br/organizacao_do_territorio/estrutura_territorial/divisao_territorial/1994/brasil.zip | `3d08a4bdbe58a8eb5e1117ef0ce6be691b2c15f6d19c00ca995f7311d060ae3b` |
| `raw/dtb/dtb_2000.zip` | territorial division / genealogy | https://geoftp.ibge.gov.br/organizacao_do_territorio/estrutura_territorial/divisao_territorial/2000/dtb_2000.zip | `54cf75d7420339d3a82d13ed719c705ce3a975b47ea70e3eb66cf38f3f907ac1` |
| `raw/dtb/dtb_2013_municipio.zip` | territorial division / genealogy | https://geoftp.ibge.gov.br/organizacao_do_territorio/estrutura_territorial/divisao_territorial/2013/dtb_2013_municipio.zip | `00ba2989f88b62f14cbb8e380955727e77636d98f3041f1a74bd10dea0504b3d` |
| `raw/dtb/DTB_2025.zip` | territorial division / genealogy | https://geoftp.ibge.gov.br/organizacao_do_territorio/estrutura_territorial/divisao_territorial/2025/DTB_2025.zip | `d077a0e48c36cf18bcc96268b4a436200c014c6b8522c1a62d894acaf39dad27` |
| `raw/ibge_evolucao/municipios_genealogia_e_alteracao_de_nome_e_grafia.pdf` | territorial division / genealogy | https://geoftp.ibge.gov.br/organizacao_do_territorio/estrutura_territorial/evolucao_da_divisao_territorial_do_brasil/evolucao_da_divisao_territorial_do_brasil_1872_2010/municipios_genealogia_e_alteracao_de_nome_e_grafia.pdf | `2050fdf18a4346e2c47ced141f2fc65b68633fb5336d29d3616066a5603711e7` |
| `raw/ibge_evolucao/notas_tecnicas.pdf` | territorial division / genealogy | https://geoftp.ibge.gov.br/organizacao_do_territorio/estrutura_territorial/evolucao_da_divisao_territorial_do_brasil/evolucao_da_divisao_territorial_do_brasil_1872_2010/notas_tecnicas.pdf | `f5ceee401d2aa6607bc197e2941d4611d410311ab743d0140971482d9057f5ae` |
| `raw/pib/base_de_dados_2002_2009_txt.zip` | PIB dos Municipios, reference-2010 series, 2002-2009 (outcome) | https://ftp.ibge.gov.br/Pib_Municipios/2022_2023/base/base_de_dados_2002_2009_txt.zip | `4ad3ba30d67f34bfcc00465cdb150d111da7bc068e7516fce61c002465dbda13` |
| `raw/pib/base_de_dados_2010_2023_txt.zip` | PIB dos Municipios, reference-2010 series, 2010-2023 (outcome) | https://ftp.ibge.gov.br/Pib_Municipios/2022_2023/base/base_de_dados_2010_2023_txt.zip | `1179047b0bcc527a776f5479a534a4b2d99d362491331b0f6b465871f9164090` |
| `raw/pib/base_1999_2012_xlsx.zip` | PIB dos Municipios, reference-2002 series, 1999-2012 (outcome) | https://ftp.ibge.gov.br/Pib_Municipios/2012/base/base_1999_2012_xlsx.zip | `b191eba79c8d7a28f62ea7fd3aaaff0fb641e9765bd42274a7c48df04e0fe61c` |
