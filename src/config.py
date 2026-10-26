"""
Central configuration: paths, file registry, column mappings, region classification,
Fokker-Planck periods, and analysis constants.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data"
DATA_CLEAN = PROJECT_ROOT / "data_clean"
RESULTS = PROJECT_ROOT / "results"
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures"


THEMED_FILES = {
    "agri_output": {
        "file": "农业产值.xlsx",
        "sheet": "原始数据",
        "skiprows": [1, 2],
        "year_col": "SgnYear",
        "key_cols": ["GrossOutputValue", "Farming", "Forestry",
                     "AnimalHusbandry", "Fishery", "FFAFServices"],
    },
    "population": {
        "file": "人口.xlsx",
        "sheet": "stata",
        "skiprows": [1, 2],
        "year_col": "SgnYear",
        "key_cols": ["PopNum"],
    },
    "gdp": {
        "file": "地区生产总值及指数.xlsx",
        "sheet": "原始数据",
        "skiprows": [1, 2],
        "year_col": "yuan",
        "key_cols": ["RegGDPIdx", "RegGDPIdx_Primary", "RegGDPIdx_Secondary",
                     "RegGDPIdx_Tertiary", "RegGDP", "RegGDP_Primary",
                     "RegGDP_Secondary", "RegGDP_Tertiary", "PerRegGDP"],
    },
    "land_area": {
        "file": "土地面积.xlsx",
        "sheet": "原始数据",
        "skiprows": [1, 2],
        "year_col": "SgnYear",
        "key_cols": ["AdmArea", "CultivateArea"],
    },
    "agri_conditions": {
        "file": "农业生产条件.xlsx",
        "sheet": "原始数据",
        "skiprows": [1, 2],
        "year_col": "SgnYear",
        "key_cols": ["AgrMachPower", "FertilizerConsm", "ElectricityConsm",
                     "IrrigatedArea"],
    },
    "income": {
        "file": "城乡居民收入.xlsx",
        "sheet": "原始数据",
        "skiprows": [1, 2],
        "year_col": "SgnYear",
        "key_cols": ["RurPerNetInc", "RurPerDisposableInc"],
    },
    "finance": {
        "file": "金融机构存贷款.xlsx",
        "sheet": "原始数据",
        "skiprows": [1, 2],
        "year_col": "SgnYear",
        "key_cols": ["Loan", "Deposit"],
    },
    "employment": {
        "file": "县域从业人员统计.xlsx",
        "sheet": "sheet1",
        "skiprows": [1, 2],
        "year_col": "SgnYear",
        "key_cols": ["EmployeeNum", "RuralEmployeeNum", "FFAFEmployeeNum"],
    },
    "spatial": {
        "file": "县域地理空间联系.xlsx",
        "sheet": "sheet1",
        "skiprows": [1, 2],
        "year_col": "SgnYear",
        "key_cols": ["CountyLng", "CountyLat", "PrefLevCityLng",
                     "PrefLevCityLat", "Instance1", "ProvCapital",
                     "ProvCapitalLng", "ProvCapitalLat", "Instance2"],
    },
    "fixed_assets": {
        "file": "县域固定资产投资.xlsx",
        "sheet": "sheet1",
        "skiprows": [1, 2],
        "year_col": "SgnYear",
        "key_cols": ["FixAst", "RealEstateDvpt", "NewAddFixAst"],
    },
    "consumption": {
        "file": "县域社会消费品零售总额.xlsx",
        "sheet": "sheet1",
        "skiprows": [1, 2],
        "year_col": "SgnYear",
        "key_cols": ["Total", "Urban", "Rural", "WholesaleRetail",
                     "HotelCatering"],
    },
    "health": {
        "file": "县域卫生条件.xlsx",
        "sheet": "sheet1",
        "skiprows": [1, 2],
        "year_col": "SgnYear",
        "key_cols": ["InstitutionNum", "PersonnelNum", "HospitalBed",
                     "MedTechPerson", "LicensedDoctor", "RegisteredNurse"],
    },
    "social_insurance": {
        "file": "社会保险与福利.xlsx",
        "sheet": "stata",
        "skiprows": [1],
        "year_col": "year",
        "key_cols": ["WelfAdoptOrgBedNum"],
    },
    "fiscal": {
        "file": "财政收支.xlsx",
        "sheet": "stata",
        "skiprows": [],
        "year_col": "year",
        "key_cols": ["FinRevenue"],
    },
    "education": {
        "file": "教育基本情况.xlsx",
        "sheet": "stata",
        "skiprows": [1],
        "year_col": "year",
        "key_cols": ["TotalEnrollment"],
    },
    "transport": {
        "file": "交通运输与邮电.xlsx",
        "sheet": "sheet1",
        "skiprows": [1, 2],
        "year_col": "SgnYear",
        "key_cols": [],
    },
}

LONG_FORMAT_FILES = {
    "crop_area": {
        "file": "农作物播种面积.xlsx",
        "sheet": "Sheet1",
        "skiprows": [1],
        "year_col": "SgnYear",
        "type_col": "CropType",
        "value_col": "SownArea",
        "filters": {
            "TotalSownArea": "农作物播种面积",
        },
    },
    "grain_area": {
        "file": "农作物播种面积.xlsx",
        "sheet": "粮食作物播种面积",
        "skiprows": [],
        "year_col": "SgnYear",
        "type_col": "CropType1",
        "value_col": "CropSownArea",
        "filters": {
            "GrainSownArea": "粮食作物",
        },
    },
    "crop_yield": {
        "file": "农产品产量.xlsx",
        "sheet": "stata",
        "skiprows": [],
        "year_col": "SgnYear",
        "type_col": "ProductType",
        "value_col": "粮食产量吨",
        "filters": {
            "GrainOutput": "粮食",
        },
    },
}

POVERTY_FILE = {
    "file": "全国832个国家级贫困县名单、摘帽整理数据.xlsx",
    "sheet": "整理数据",
    "col_map": {
        "行政区划代码": "CountyCode",
        "贫困县": "CountyName",
        "所属地域": "Region",
        "所属省份": "Province",
        "所属城市": "City",
        "摘帽时间": "ExitYear",
    },
    "expected_rows": 832,
}

MAIN_DATABASE = {
    "file": "中国县域数据库6.0版_0514090132.xlsx",
    "sheets": {
        "raw": "原始数据",
        "interpolated": "线性插值",
    },
    "year_col": "年份",
    "county_code_col": "区县代码",
    "col_map": {
        "年份": "Year",
        "省份": "Province",
        "城市": "City",
        "区县": "CountyName",
        "区县代码": "CountyCode",
        "行政区域土地面积_平方公里": "AdmArea_km2",
        "年末总人口_万人": "PopTotal",
        "户籍人口数_万人": "PopNum",
        "乡村人口_万人": "RuralPop",
        "乡村从业人员数_人": "RuralEmployeeNum",
        "农林牧渔业从业人员数_人": "FFAFEmployeeNum",
    },
}

COL = {
    "Y_agri": "GrossOutputValue",
    "Y_nonagri": "NonAgriGDP",
    "N": "PopNum",
    "A": "CultivateArea",
    "GDP": "RegGDP",
    "GDP_primary": "RegGDP_Primary",
    "GDP_secondary": "RegGDP_Secondary",
    "GDP_tertiary": "RegGDP_Tertiary",
    "PerGDP": "PerRegGDP",
    "year": "Year",
    "county_code": "CountyCode",
    "province": "Province",
    "city": "City",
    "county_name": "CountyName",
    "is_poverty": "IsPoverty",
    "exit_year": "ExitYear",
    "lng": "CountyLng",
    "lat": "CountyLat",
    "region": "Region",
}

EAST_PROVINCES = [
    "北京市", "天津市", "河北省", "辽宁省", "上海市", "江苏省",
    "浙江省", "福建省", "山东省", "广东省", "海南省",
]
CENTRAL_PROVINCES = [
    "山西省", "吉林省", "黑龙江省", "安徽省", "江西省",
    "河南省", "湖北省", "湖南省",
]
WEST_PROVINCES = [
    "内蒙古自治区", "广西壮族自治区", "重庆市", "四川省", "贵州省",
    "云南省", "西藏自治区", "陕西省", "甘肃省", "青海省",
    "宁夏回族自治区", "新疆维吾尔自治区",
]

_PROVINCE_TO_REGION = {}
for p in EAST_PROVINCES:
    _PROVINCE_TO_REGION[p] = "East"
for p in CENTRAL_PROVINCES:
    _PROVINCE_TO_REGION[p] = "Central"
for p in WEST_PROVINCES:
    _PROVINCE_TO_REGION[p] = "West"


def get_region(province: str) -> str:
    """Map province name to East / Central / West."""
    return _PROVINCE_TO_REGION.get(province, "Unknown")


PROVINCE_CODE_MAP = {
    11: "北京市", 12: "天津市", 13: "河北省", 14: "山西省",
    15: "内蒙古自治区", 21: "辽宁省", 22: "吉林省", 23: "黑龙江省",
    31: "上海市", 32: "江苏省", 33: "浙江省", 34: "安徽省",
    35: "福建省", 36: "江西省", 37: "山东省", 41: "河南省",
    42: "湖北省", 43: "湖南省", 44: "广东省", 45: "广西壮族自治区",
    46: "海南省", 50: "重庆市", 51: "四川省", 52: "贵州省",
    53: "云南省", 54: "西藏自治区", 61: "陕西省", 62: "甘肃省",
    63: "青海省", 64: "宁夏回族自治区", 65: "新疆维吾尔自治区",
}

FP_PERIODS = {
    "pre":    (2000, 2012),
    "during": (2013, 2019),
    "post":   (2020, 2023),
}

CRISIS_YEARS = frozenset({2008, 2009, 2020, 2021, 2022})

BASE_YEAR = 2000
N_BINS_FP = 25
BOOTSTRAP_N = 1000
MIN_BIN_COUNT = 30
YEAR_RANGE = (2000, 2023)

DEFLATE_VARS = [
    "GrossOutputValue", "Farming", "Forestry", "AnimalHusbandry",
    "Fishery", "FFAFServices",
    "RegGDP", "RegGDP_Primary", "RegGDP_Secondary", "RegGDP_Tertiary",
    "PerRegGDP",
    "RurPerNetInc", "RurPerDisposableInc",
    "Loan", "Deposit",
    "FixAst",
]

SCALING_EQUATIONS = {
    "14a": {
        "label": "Non-agricultural GDP ~ Population",
        "y": "NonAgriGDP",
        "x": ["PopNum"],
        "expected_sign": "beta > 1 (superlinear)",
    },
    "14b": {
        "label": "Agricultural output ~ Population",
        "y": "GrossOutputValue",
        "x": ["PopNum"],
        "expected_sign": "beta < 1 (sublinear)",
    },
    "14c": {
        "label": "Agricultural output ~ Population + CultivateArea",
        "y": "GrossOutputValue",
        "x": ["PopNum", "CultivateArea"],
        "expected_sign": "beta_N < beta_14b (controlling for land reveals deeper sublinearity — crowding effect)",
    },
    "14d": {
        "label": "Non-agricultural GDP ~ Population + CultivateArea",
        "y": "NonAgriGDP",
        "x": ["PopNum", "CultivateArea"],
        "expected_sign": "beta_N ≈ beta_14a (land irrelevant)",
    },
    "14e": {
        "label": "Agricultural output ~ CultivateArea",
        "y": "GrossOutputValue",
        "x": ["CultivateArea"],
        "expected_sign": "gamma > 0 (land matters)",
    },
    "14f": {
        "label": "Agricultural output ~ TotalSownArea",
        "y": "GrossOutputValue",
        "x": ["TotalSownArea"],
        "expected_sign": "gamma > 0 (sown area as flow measure of land)",
    },
    "14g": {
        "label": "Agricultural output ~ PopNum + TotalSownArea",
        "y": "GrossOutputValue",
        "x": ["PopNum", "TotalSownArea"],
        "expected_sign": "beta_N < beta_14b (alternative land control)",
    },
    "14h": {
        "label": "Secondary GDP ~ PopNum",
        "y": "RegGDP_Secondary",
        "x": ["PopNum"],
        "expected_sign": "beta ≈ 1 (near-linear, industrial)",
    },
    "14i": {
        "label": "Tertiary GDP ~ PopNum",
        "y": "RegGDP_Tertiary",
        "x": ["PopNum"],
        "expected_sign": "beta > 1 (superlinear, knowledge-intensive)",
    },
    "14j": {
        "label": "Primary GDP ~ PopNum",
        "y": "RegGDP_Primary",
        "x": ["PopNum"],
        "expected_sign": "beta < 1 (diminishing returns)",
    },
}
