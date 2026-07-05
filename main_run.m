% 清理环境
clear;
clc;

%% ===================== 1. 算法大向量输入区 (35×1) =====================
% 在这里定义遗传算法或粒子群算法生成的个体向量 X
% 下面以第一问的数据作为测试大向量输入（其余不用的弹延迟设为9999）：
X = [
    136.4375;
    113.218670811;
    120;
    120;
    120;
    3.11808532503;
    -1.36882790713;
    2.65163532734;
    -2.97939382013;
    2.97395021576;
    0;
    6.83239366912;
    0;
    0;
    0;
    0;
    0;
    0;
    0;
    0;
    0;
    0;
    0;
    0;
    0;
    3.6;
    1;
    1;
    1;
    1;
    1;
    1;
    1;
    1;
    1;
    1;
    1;
    1;
    1;
    1
];

%% ===================== 2. 大向量自动分割与解包 =====================
UAV_v      = X(1:5);
UAV_phi    = X(6:10);
t_first    = X(11:15);
dt12_extra = X(16:20);
dt23_extra = X(21:25);
% 将 15x1 的引信延迟还原为 5x3 矩阵
t_delay_matrix = reshape(X(26:40), 3, 5).'; 

%% ===================== 3. 约束条件逻辑检查与惩罚机制 =====================
is_valid = true;
% (1) 速度限制：70 <= v <= 140
if any(UAV_v < 70) || any(UAV_v > 140), is_valid = false; end
% (2) 角度限制：-pi <= phi <= pi
if any(UAV_phi < -pi) || any(UAV_phi > pi), is_valid = false; end
% (3) 投放间隔限制：额外间隔必须大于等于0
if any(dt12_extra < 0) || any(dt23_extra < 0), is_valid = false; end
% (4) 时间不能为负
if any(t_first < 0) || any(t_delay_matrix < 0, 'all'), is_valid = false; end

%% ===================== 4. 自动生成投放时间与参数封装 =====================
if is_valid
    % 计算实际间隔
    dt12 = 1 + dt12_extra;
    dt23 = 1 + dt23_extra;
    
    t_drop_matrix = zeros(5,3);
    for i = 1:5
        t_drop_matrix(i,1) = t_first(i);
        t_drop_matrix(i,2) = t_drop_matrix(i,1) + dt12(i);
        t_drop_matrix(i,3) = t_drop_matrix(i,2) + dt23(i);
    end
    % 封装无人机参数
    UAV_params = struct('v',{},'theta',{},'phi',{});
    for i = 1:5
        UAV_params(i).v = UAV_v(i);
        UAV_params(i).theta = 0;      
        UAV_params(i).phi = UAV_phi(i);
    end
    % 封装烟幕弹参数
    Bomb_params(5,3) = struct('is_active',false,'t_drop',0,'t_explode',0);
    for i = 1:5
        for k = 1:3
            Bomb_params(i,k).is_active = true;
            Bomb_params(i,k).t_drop = t_drop_matrix(i,k);
            Bomb_params(i,k).t_explode = t_drop_matrix(i,k) + t_delay_matrix(i,k);
        end
    end
    
    %% ===================== 5. 调用物理模型并启动高级计时 =====================
    tic; % 开始计时
    [total_time, missile_times] = smoke_interference_model(UAV_params, Bomb_params);
    runtime = toc; % 结束计时
    
else
    total_time = 0;
    missile_times = [0, 0, 0];
    runtime = 0;
    warning('当前输入的大向量 X 违反了边界约束约束，仿真结果强行归零。');
end

%% ===================== 6. 输出结果 =====================
fprintf('\n');
fprintf('================== 仿真结果 ==================\n');
fprintf('总有效遮蔽时间：%.2f s\n', total_time);
fprintf('----------------------------------------------\n');
fprintf('导弹 M1：%.2f s\n', missile_times(1));
fprintf('导弹 M2：%.2f s\n', missile_times(2));
fprintf('导弹 M3：%.2f s\n', missile_times(3));
fprintf('----------------------------------------------\n');
fprintf('模型单次运行耗时：%.6f 秒\n', runtime);
fprintf('==============================================\n');