function [total_time, missile_times, runtime] = run_smoke_simulation(x)
% RUN_SMOKE_SIMULATION 烟幕干扰模型仿真函数
%   输入：x - 40维向量
%   输出：total_time   - 总有效遮蔽时间（秒）
%         missile_times - 三枚导弹的遮蔽时间 [M1, M2, M3]（秒）
%         runtime      - 模型单次运行耗时（秒）

    %% 1. 解包40维向量
    UAV_v      = x(1:5);
    UAV_phi    = x(6:10);
    t_first    = x(11:15);
    dt12_extra = x(16:20);
    dt23_extra = x(21:25);
    t_delay_matrix = reshape(x(26:40), 3, 5).'; 

    %% 2. 约束检查
    is_valid = true;
    if any(UAV_v < 70) || any(UAV_v > 140), is_valid = false; end
    if any(UAV_phi < -pi) || any(UAV_phi > pi), is_valid = false; end
    if any(dt12_extra < 0) || any(dt23_extra < 0), is_valid = false; end
    if any(t_first < 0) || any(t_delay_matrix < 0, 'all'), is_valid = false; end

    %% 3. 如果有效则运行仿真
    if is_valid
        % 计算投放时间
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
        
        % 运行仿真
        tic;
        [total_time, missile_times] = smoke_interference_model(UAV_params, Bomb_params);
        runtime = toc;
        
    else
        total_time = 0;
        missile_times = [0, 0, 0];
        runtime = 0;
    end

end
