# from types import SimpleNamespace


# # ---------------------#
# #   Knapsack Problem   #
# # ---------------------#

# kp = SimpleNamespace(
#     # type of data (general or instance)
#     data_type = "general",

#     # variable parameters between isntances
#     n_items = [20, 30, 40, 50, 60, 70, 80],
#     correlation = ["UN", "WC", "ASC", "SC"],
#     h = [40, 80],
#     delta = [0.1, 0.5, 1.0],
#     budget_factor = [0.1, 0.15, 0.20],

#     # fixed parameters between isntances
#     R = 1000,
#     H = 100,

#     # data generation parameters
#     time_limit = 30,            # for data generation only
#     mip_gap = 0.01,             # for data generation only
#     verbose = 0,                # for data generation only
#     threads = 1,                # for data generation only
#     tr_split=0.80,              # train/test split size
    
#     # n_samples_inst = 500,       # number of instances to samples
#     # n_samples_fs = 10,          # number of first-stage decisions samples per problem
#     # n_samples_per_fs = 50,      # number of uncertainty samples per first-stage decision

#     n_samples_inst = 50,       # number of instances to samples
#     n_samples_fs = 5,          # number of first-stage decisions samples per problem
#     n_samples_per_fs = 10,      # number of uncertainty samples per first-stage decision



#     # generic parameters
#     seed = 7,
#     data_path = './data/',

# )


# # --------------------------- #
# #   Capital Budgeting Problem #
# # --------------------------- #

# cb = SimpleNamespace(
#     # problem parameters
#     n_items=[10, 20, 30, 40, 50],
#     k=.8,
#     loans=0,
#     l=.12,    # default but not needed for this instance
#     m=1.2,     # default but also not needed
#     xi_dim=4,

#     # data generation parameters
#     obj_label="fs_plus_ss_obj",     # label for objective prediction
#     feas_label="min_budget_cost",   # label for feasibility prediction
    
#     time_limit=30,  # for data generation only
#     mip_gap=0.01,   # for data generation only
#     verbose=0,      # for data generation only
#     threads=1,      # for data generation only

#     tr_split=0.80,  # train/test split size

#     # n_samples_inst = 500,       # number of instances to samples
#     # n_samples_fs = 10,          # number of first-stage decisions samples per problem
#     # n_samples_per_fs = 50,      # number of uncertainty samples per first-stage decision

#     n_samples_inst = 50,       # number of instances to samples
#     n_samples_fs = 5,          # number of first-stage decisions samples per problem
#     n_samples_per_fs = 10,      # number of uncertainty samples per first-stage decision


#     # generic parameters
#     seed=1,
#     inst_seed=range(1, 101),
#     data_path='./data/',
# )



# # ----------------------------------#
# # Offering strategy (no network)     #
# # ----------------------------------#
# offering_no_network = SimpleNamespace(
#     # -------- 基本尺寸/数据来源 --------
#     data_type="instance",          # 该问题更像“固定一个实例”（价格矩阵+负荷+PV边界）
#     Sbase=1000.0,                  # 与Matlab一致：kW/kWh -> pu（除以Sbase）
#     T=24,
#     n_scenarios=25,
#     price_mat_path="./data/offering_no_network/price_matrix_revised_1000.mat",
#     lambda_rt_value=800.0,

#     # DA 价格矩阵（Matlab: price_matrix_revised_100.mat, 24x25）
#     price_mat_file="./data/offering_no_network/price_matrix_revised_1000.mat",
#     price_mat_var="price_matrix_revised_1000",

#     # 概率（Matlab MP: uniform）
#     rho_mode="uniform",

#     # RT 偏差惩罚价（Matlab：lambda_rt = 800 * ones(24,S)）
#     rt_price_mode="constant",
#     rt_price_const=800.0,

#     # -------- PV 不确定性（与Matlab一致：min/max + shift + Gamma预算）--------
#     pv_min_kw=[240.0 * v for v in [
#         0,0,0,0,0,0,0, 1.98275498,2.40239236,3.68092735,5.03178247,5.03178247,
#         5.03178247,5.03178247,5.03178247,5.03178247,4.27448781,3.34392749,
#         2.12267122,1.69047437,0,0,0,0
#     ]],

#     pv_max_kw=[240.0 * v for v in [
#         0,0,0,0,0,0,0, 5.29490001,5.71453739,6.99307238,8.34392749,8.34392749,
#         8.34392749,8.34392749,8.34392749,8.34392749,7.58663283,6.65607251,
#         5.43481625,5.00261939,0,0,0,0
#     ]],

#     pv_max_shift_kw=1.0,           # Matlab: p_pv_max = p_pv_max + 1*ones(1,24)
#     Gamma=12.0,

#     # -------- 聚合负荷（Matlab示例：10*[...] / Sbase）--------
#     load_sum_kw=[10.0 * v for v in [
#         111.3822392,101.5065653,94.30973857,93.17348758,94.48769219,99.68165177,
#         112.8326054,116.1722534,124.5485385,131.4933456,143.5932615,160.0526555,
#         167.6674216,182.9392865,203.4062827,221.0867556,239.826844,244.9674648,
#         232.526104,211.0252955,198.7750239,168.5545511,142.6519859,122.7069434
#     ]],

#     # -------- ESS 参数（与Matlab一致）--------
#     N_es=10,
#     E_ess_per_unit_kwh=14.5,       # Matlab: E_ess_0 = N_es*14.5/Sbase
#     P_ess_per_unit_kw=11.3,        # Matlab: P_ess_max_0 = N_es*11.3/Sbase
#     eta=0.95,

#     # 成本（Matlab示例保持同量级）
#     ESS_cost=1e-6,                 # 你也可以按Matlab写法在建模处换算
#     PV_cost=1e-6,

#     # --------------------------------------------------------------------
#     # 关键：保留 offer curve “单调性”采样方式（贴近Matlab MP 的单调性约束）
#     # Matlab 约束：if lambda_da(t,s) > lambda_da(t,sp) => p_DA(t,s) >= p_DA(t,sp)
#     # --------------------------------------------------------------------
#     offer_curve_sampling="monotone_by_price",  # 必须保留
#     # 为了“最小可跑”，需要给采样一个合理范围（pu）
#     # 下面 margin 用来放宽边界，避免过窄导致样本分布奇怪
#     p_da_bound_margin_pu=0.05,

#     # -------- 数据集生成（最小可跑）--------
#     time_limit=30,
#     mip_gap=0.02,
#     verbose=0,
#     threads=1,
#     tr_split=0.80,
#     n_samples_inst=50,              # 最小可跑：1个实例
#     n_samples_fs=10,                # 最小可跑：每个实例采2条offer curve
#     n_samples_per_fs=50,            # 最小可跑：每条curve采3个不确定性xi

#     # -------- 通用 --------
#     seed=7,
#     data_path="./data/",
# )






# /content/drive/MyDrive/Neur2RO/ro/params.py
from types import SimpleNamespace


# ---------------------#
#   Knapsack Problem   #
# ---------------------#

kp = SimpleNamespace(
    # type of data (general or instance)
    data_type="general",

    # variable parameters between isntances
    n_items=[20, 30, 40, 50, 60, 70, 80],
    correlation=["UN", "WC", "ASC", "SC"],
    h=[40, 80],
    delta=[0.1, 0.5, 1.0],
    budget_factor=[0.1, 0.15, 0.20],

    # fixed parameters between isntances
    R=1000,
    H=100,

    # data generation parameters
    time_limit=30,            # for data generation only
    mip_gap=0.01,             # for data generation only
    verbose=0,                # for data generation only
    threads=1,                # for data generation only
    tr_split=0.80,            # train/test split size

    n_samples_inst=50,        # number of instances to samples
    n_samples_fs=5,           # number of first-stage decisions samples per problem
    n_samples_per_fs=10,      # number of uncertainty samples per first-stage decision

    # generic parameters
    seed=7,
    data_path="./data/",
)


# --------------------------- #
#   Capital Budgeting Problem #
# --------------------------- #

cb = SimpleNamespace(
    # problem parameters
    n_items=[10, 20, 30, 40, 50],
    k=.8,
    loans=0,
    l=.12,    # default but not needed for this instance
    m=1.2,    # default but also not needed
    xi_dim=4,

    # data generation parameters
    obj_label="fs_plus_ss_obj",     # label for objective prediction
    feas_label="min_budget_cost",   # label for feasibility prediction

    time_limit=30,  # for data generation only
    mip_gap=0.01,   # for data generation only
    verbose=0,      # for data generation only
    threads=1,      # for data generation only

    tr_split=0.80,  # train/test split size

    n_samples_inst=50,       # number of instances to samples
    n_samples_fs=5,          # number of first-stage decisions samples per problem
    n_samples_per_fs=10,     # number of uncertainty samples per first-stage decision

    # generic parameters
    seed=1,
    inst_seed=range(1, 101),
    data_path="./data/",
)


# ----------------------------------#
# Offering strategy (no network)     #
# ----------------------------------#
offering_no_network = SimpleNamespace(
    # -------- 基本尺寸/数据来源 --------
    data_type="instance",
    Sbase=1000.0,
    T=24,
    n_scenarios=25,
    price_mat_path="./data/offering_no_network/price_matrix_revised_1000.mat",
    lambda_rt_value=800.0,

    price_mat_file="./data/offering_no_network/price_matrix_revised_1000.mat",
    price_mat_var="price_matrix_revised_1000",

    rho_mode="uniform",

    rt_price_mode="constant",
    rt_price_const=800.0,

    # -------- PV 不确定性（与Matlab一致：min/max + shift + Gamma预算）--------
    pv_min_kw=[240.0 * v for v in [
        0,0,0,0,0,0,0, 1.98275498,2.40239236,3.68092735,5.03178247,5.03178247,
        5.03178247,5.03178247,5.03178247,5.03178247,4.27448781,3.34392749,
        2.12267122,1.69047437,0,0,0,0
    ]],

    pv_max_kw=[240.0 * v for v in [
        0,0,0,0,0,0,0, 5.29490001,5.71453739,6.99307238,8.34392749,8.34392749,
        8.34392749,8.34392749,8.34392749,8.34392749,7.58663283,6.65607251,
        5.43481625,5.00261939,0,0,0,0
    ]],

    pv_max_shift_kw=1.0,
    Gamma=12.0,

    # -------- 聚合负荷（Matlab示例：10*[...] / Sbase）--------
    load_sum_kw=[10.0 * v for v in [
        111.3822392,101.5065653,94.30973857,93.17348758,94.48769219,99.68165177,
        112.8326054,116.1722534,124.5485385,131.4933456,143.5932615,160.0526555,
        167.6674216,182.9392865,203.4062827,221.0867556,239.826844,244.9674648,
        232.526104,211.0252955,198.7750239,168.5545511,142.6519859,122.7069434
    ]],

    # -------- ESS 参数（与Matlab一致）--------
    N_es=10,
    E_ess_per_unit_kwh=14.5,
    P_ess_per_unit_kw=11.3,
    eta=0.95,

    ESS_cost=1e-6,
    PV_cost=1e-6,

    offer_curve_sampling="monotone_by_price",
    p_da_bound_margin_pu=0.05,

    time_limit=30,
    mip_gap=0.02,
    verbose=0,
    threads=1,
    tr_split=0.80,
    n_samples_inst=50,
    n_samples_fs=10,
    n_samples_per_fs=50,

    seed=7,
    data_path="./data/",
)


# ----------------------------------#
# Offering strategy (with network)   #
# ----------------------------------#
offering_network = SimpleNamespace(
    # 基本
    data_type="instance",
    T=24,

    # 注意：这里 Sbase 代表你网络数据/功率单位的标幺基准（与 .mat 里一致）
    # 若你的网络数据以 100kVA 为基准，用 100；若以 1000kW/kWh，用 1000
    Sbase=100.0,

    # 价格矩阵（24 x S）
    n_scenarios=25,
    price_mat_path="./data/offering_network/price_matrix_revised_1000.mat",
    price_mat_var=None,
    rho_mode="uniform",

    # RT 偏差价格（可以先用常数）
    lambda_rt_value=800.0,

    # 网络数据（.mat）
    # 需要包含：A, A0, RD, XD（或至少 RD, XD, A, A0）
    ldf_mat_path="./data/offering_network/123_LDF.mat",
    bus_phase_mat_path="./data/offering_network/load_name_bus_phase.mat",
    load_mat_path="./data/offering_network/123_load.mat",

    # DER bus 名称（与你 matlab 一致，用于在 load_name_bus_phase 里定位索引）
    DER_BusName=[
        "S1a","S1b","S1c","S29a","S29b","S29c","S35a","S35b","S35c",
        "S44a","S44b","S44c","S51a","S51b","S51c","S86a","S86b","S86c",
        "S108a","S108b","S108c"
    ],

    # 电压上下界（电压平方）
    V_min=0.90,
    V_max=1.10,

    # 24小时负荷分配系数（与你 matlab 概率向量一致）
    load_hour_factors=[
        0.026503717, 0.028398378, 0.032184586, 0.035970794, 0.037862073, 0.041649898,
        0.043541177, 0.045432456, 0.049218664, 0.053004872, 0.054896151, 0.056787429,
        0.053004872, 0.049218664, 0.045432456, 0.039753352, 0.037862073, 0.041649898,
        0.045432456, 0.053004872, 0.049218664, 0.041649898, 0.034076774, 0.030290566
    ],

    # PV 不确定性：先按“同一小时全 DER 共享 xi(t)”的相关不确定性（维度仍为 24）
    # PV min/max（按小时），会自动复制到 n_der 个 DER
    pv_min_kw=[0.0]*7 + [240.0*v for v in [1.98275498,2.40239236,3.68092735,5.03178247,5.03178247,5.03178247,5.03178247,5.03178247,5.03178247,4.27448781,3.34392749,2.12267122,1.69047437]] + [0.0]*4,
    pv_max_kw=[0.0]*7 + [240.0*v for v in [5.29490001,5.71453739,6.99307238,8.34392749,8.34392749,8.34392749,8.34392749,8.34392749,8.34392749,7.58663283,6.65607251,5.43481625,5.00261939]] + [0.0]*4,
    pv_max_shift_kw=1.0,
    Gamma=12.0,

    # Flexible load（可选）：若你暂时不想用 DR/Fload，可把 enable_fload=False
    enable_fload=True,
    # 7x24 原始 Fload_0_orig（kW），会 kron 到 21x24，并按你的 matlab 方式做 -2kW 留最小负荷
    fload_0_orig_kw=[
        [28.403,27.745,25.595,28.858,27.907,38.154,35.231,41.2,33.813,37.836,44.419,43.83,41.183,40.113,38.614,44.702,52.482,53.223,44.659,42.954,42.16,33.82,27.996,25.726],
        [26.708,24.858,25.325,26.854,27.985,39.732,38.573,38.658,32.45,30.887,30.881,37.759,41.552,34.551,40.839,50.903,50.605,46.766,51.589,49.475,41.592,32.965,29.287,24.987],
        [26.12,23.628,25.965,23.224,24.335,28.837,36.887,34.475,34.442,35.13,29.114,33.104,32.029,36.393,30.638,38.503,48.347,46.176,46.177,44.766,40.912,35.985,27.62,26.049],
        [25.179,23.709,23.727,29.5,28.324,31.996,37.915,40.539,33.414,34.971,36.674,39.532,33.595,37.064,35.727,47.89,52.252,50.777,53.641,55.291,43.828,42.018,33.738,29.795],
        [33.249,26.929,27.074,26.424,30.609,39.057,34.964,36.167,48.314,39.416,42.604,41.926,45.965,40.968,41.437,45.023,47.052,43.705,36.877,39.688,38.824,37.688,32.84,28.788],
        [31.949,27.441,26.953,26.404,30.105,33.339,35.819,40.983,37.922,42.707,46.129,48.732,48.593,46.93,49.832,57.155,55.125,52.432,54.479,45.009,36.146,34.71,35.04,30.189],
        [27.27,26.902,26.891,24.765,27.193,30.024,35.482,45.443,49.897,53.75,50.579,60.073,64.758,56.226,68.008,61.09,59.055,57.432,52.781,56.064,45.286,39.492,33.21,29.44],
    ],
    fload_keep_min_kw=2.0,          # 你的 matlab：Fload_adj = Fload_0 - 2
    fload_scale=0.2,               # 你的 matlab：Fload_0_orig 前乘 0.2
    fload_cost_vec=[1,1.5,1.2,2,1.8,1.4,2.2],  # 7维 costVec，会 kron 到 21 个 DER（3相复制）

    # ESS（按 DER 分布，每个 DER 都一套）
    E_ess_per_unit_kwh=14.5,
    P_ess_per_unit_kw=11.3,
    eta=0.95,
    soc0_frac=0.5,                 # 初始 SOC = 0.5 * Emax

    # 成本
    ESS_cost=1e-3,                 # 先给一个更显著的数，避免训练标签太小（可改回 1e-6）
    PV_cost=1e-6,
    Fload_cost_scale=1.0,          # 如需整体缩放

    # 训练采样：仍保留 monotone_by_price（在数据生成层面）
    offer_curve_sampling="monotone_by_price",
    p_da_bound_margin_pu=0.05,

    # 数据生成参数（网络 MIP 通常更慢：建议 time_limit 大一点）
    time_limit=120,
    mip_gap=0.02,
    verbose=0,
    threads=1,
    tr_split=0.80,
    n_samples_inst=25,
    n_samples_fs=2,
    n_samples_per_fs=2,

    seed=7,
    data_path="./data/",
)


