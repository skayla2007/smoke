function [total_cover_time, missile_cover_times] = smoke_interference_model(UAV_params, Bomb_params)
    [total_cover_time, missile_cover_times] = smoke_interference_model_interval(UAV_params, Bomb_params);
    %{
% SMOKE_INTERFERENCE_MODEL 极致优化版（区间驱动+向量化全遮挡）

    %% 1. 固定参数设定
    g = 9.8; 
    v_m = 300;
    r_cloud_sq = 10^2; % 预计算平方，避免循环内norm开方
    life_cloud = 20;
    v_cloud_down = 3;
    
    M_init = [20000, 0, 2000; 19000, 600, 2100; 18000, -600, 1900];
    num_missiles = 3;
    
    UAV_init = [17800, 0, 1800; 12000, 1400, 1400; 6000, -3000, 700; 11000, 2000, 1800; 13000, -2000, 1300];
    num_uavs = 5;
    
    t_m_max = [norm(M_init(1,:)); norm(M_init(2,:)); norm(M_init(3,:))] / v_m;

    %% 2. 真目标圆柱体采样点生成 (200x3 矩阵)
    target_center_bottom = [0, 200, 0];
    target_r = 7; target_h = 10; N_samples = 100;
    angles = linspace(0, 2*pi, N_samples+1).'; angles(end) = []; 
    cos_a = target_r * cos(angles); sin_a = target_r * sin(angles);
    target_pts = [
        target_center_bottom(1) + cos_a, target_center_bottom(2) + sin_a, zeros(N_samples, 1);
        target_center_bottom(1) + cos_a, target_center_bottom(2) + sin_a, target_h * ones(N_samples, 1)
    ];
    num_pts = 200;

    %% 3. 提取所有烟幕弹的生命周期区间与起爆点
    cloud_t_start = zeros(15, 1);
    cloud_t_end   = zeros(15, 1);
    cloud_pos_exp = zeros(15, 3);
    idx = 0;
    
    for i = 1:num_uavs
        % 考虑决策大向量中 theta 恒为 0，简化速度向量计算
        v_vec = UAV_params(i).v * [cos(UAV_params(i).phi), sin(UAV_params(i).phi), 0];
        for k = 1:3
            if Bomb_params(i, k).is_active
                t_d = Bomb_params(i, k).t_drop;
                t_e = Bomb_params(i, k).t_explode;
                t_fall = t_e - t_d;
                if t_fall < 0, continue; end % 非法弹跳过
                
                idx = idx + 1;
                pos_drop = UAV_init(i, :) + v_vec * t_d;
                cloud_pos_exp(idx, :) = pos_drop + v_vec * t_fall - [0, 0, 0.5 * g * t_fall^2];
                cloud_t_start(idx) = t_e;
                cloud_t_end(idx)   = t_e + life_cloud;
            end
        end
    end
    cloud_t_start(idx+1:end) = []; cloud_t_end(idx+1:end) = []; cloud_pos_exp(idx+1:end, :) = [];
    num_clouds = idx;

    % 如果没有任何有效烟幕弹，直接返回 0 结束，实现瞬间退弹
    if num_clouds == 0
        total_cover_time = 0; missile_cover_times = [0,0,0]; return;
    end

    %% 4. 区间驱动策略：只在“可能被遮挡的时间段”内高精离散
    % 任何有意义的遮挡只能发生在某些烟幕弹存在的时间段内
    min_t = min(cloud_t_start);
    max_t = min(max(cloud_t_end), max(t_m_max));
    
    if min_t > max_t
        total_cover_time = 0; missile_cover_times = [0,0,0]; return;
    end

    % 0.01s 步长的高精仿真只限制在 [min_t, max_t] 这一战略核心期
    dt = 0.01; 
    time_steps = min_t : dt : max_t;
    len_steps = length(time_steps);
    missile_is_covered = false(len_steps, num_missiles);

    M_dirs = zeros(3, 3);
    for j = 1:num_missiles
        M_dirs(j, :) = -M_init(j, :) / norm(M_init(j, :));
    end

    %% 5. 极速向量化碰撞核验
    for s = 1:len_steps
        t = time_steps(s);
        
        % 提取当前时刻活跃的烟幕中心
        active_idx = (t >= cloud_t_start) & (t <= cloud_t_end);
        if ~any(active_idx), continue; end 
        
        dt_active = t - cloud_t_start(active_idx);
        active_centers = cloud_pos_exp(active_idx, :);
        active_centers(:, 3) = active_centers(:, 3) - v_cloud_down * dt_active;
        num_active = size(active_centers, 1);
        
        for j = 1:num_missiles
            if t > t_m_max(j), continue; end 
            
            pos_M = M_init(j, :) + (v_m * t) * M_dirs(j, :);
            all_pts_blocked = true; 
            
            % 优化核心：不再为每个点做复杂的单循环，而是直接展开数学判定
            for p = 1:num_pts
                pos_P = target_pts(p, :); 
                v_MP = pos_P - pos_M;
                len_MP_sq = v_MP(1)^2 + v_MP(2)^2 + v_MP(3)^2;
                
                pt_p_is_blocked = false;
                for c = 1:num_active
                    % 计算线段投影系数 u
                    v_MC_1 = active_centers(c,1) - pos_M(1);
                    v_MC_2 = active_centers(c,2) - pos_M(2);
                    v_MC_3 = active_centers(c,3) - pos_M(3);
                    
                    u = (v_MC_1*v_MP(1) + v_MC_2*v_MP(2) + v_MC_3*v_MP(3)) / len_MP_sq;
                    if u < 0, u = 0; elseif u > 1, u = 1; end
                    
                    % 纯标量数学计算，避开所有矩阵开销，计算距离平方值
                    c_pt_1 = pos_M(1) + u * v_MP(1);
                    c_pt_2 = pos_M(2) + u * v_MP(2);
                    c_pt_3 = pos_M(3) + u * v_MP(3);
                    
                    dist_sq = (c_pt_1 - active_centers(c,1))^2 + ...
                              (c_pt_2 - active_centers(c,2))^2 + ...
                              (c_pt_3 - active_centers(c,3))^2;
                    
                    if dist_sq <= r_cloud_sq
                        pt_p_is_blocked = true; 
                        break; 
                    end
                end
                
                % 提前短路中止逻辑
                if ~pt_p_is_blocked
                    all_pts_blocked = false; 
                    break; 
                end
            end
            missile_is_covered(s, j) = all_pts_blocked;
        end
    end

    %% 6. 计算最终遮蔽时长
    missile_cover_times = sum(missile_is_covered, 1) * dt;
    total_cover_time = sum(any(missile_is_covered, 2)) * dt;
    %}
end

function [total_cover_time, missile_cover_times] = smoke_interference_model_interval(UAV_params, Bomb_params)
%SMOKE_INTERFERENCE_MODEL_INTERVAL Interval-driven optimized implementation.
% Single-cloud full-cover intervals are found by bisection. Multi-cloud
% checks are limited to windows where two or more single-cloud partial
% intervals overlap.

    %% 1. Fixed parameters
    g = 9.8;
    v_m = 300;
    r_cloud_sq = 10^2;
    life_cloud = 20;
    v_cloud_down = 3;

    dt_multi = 0.01;
    dt_bracket = 0.05;
    tol_time = 1e-5;

    M_init = [20000, 0, 2000; 19000, 600, 2100; 18000, -600, 1900];
    num_missiles = size(M_init, 1);

    UAV_init = [17800, 0, 1800; 12000, 1400, 1400; 6000, -3000, 700; 11000, 2000, 1800; 13000, -2000, 1300];
    num_uavs = size(UAV_init, 1);
    num_bombs_per_uav = size(Bomb_params, 2);

    t_m_max = zeros(num_missiles, 1);
    M_dirs = zeros(num_missiles, 3);
    for j = 1:num_missiles
        t_m_max(j) = norm(M_init(j, :)) / v_m;
        M_dirs(j, :) = -M_init(j, :) / norm(M_init(j, :));
    end

    %% 2. Target cylinder sampling
    target_center_bottom = [0, 200, 0];
    target_r = 7;
    target_h = 10;
    N_samples = 100;

    angles = linspace(0, 2*pi, N_samples + 1).';
    angles(end) = [];
    cos_a = target_r * cos(angles);
    sin_a = target_r * sin(angles);

    target_pts = [
        target_center_bottom(1) + cos_a, target_center_bottom(2) + sin_a, zeros(N_samples, 1);
        target_center_bottom(1) + cos_a, target_center_bottom(2) + sin_a, target_h * ones(N_samples, 1)
    ];
    num_pts = size(target_pts, 1);

    %% 3. Build cloud life intervals and explosion positions
    max_clouds = num_uavs * num_bombs_per_uav;
    cloud_t_start = zeros(max_clouds, 1);
    cloud_t_end = zeros(max_clouds, 1);
    cloud_pos_exp = zeros(max_clouds, 3);
    idx = 0;

    for i = 1:num_uavs
        v_vec = UAV_params(i).v * [cos(UAV_params(i).phi), sin(UAV_params(i).phi), 0];
        for k = 1:num_bombs_per_uav
            if Bomb_params(i, k).is_active
                t_d = Bomb_params(i, k).t_drop;
                t_e = Bomb_params(i, k).t_explode;
                t_fall = t_e - t_d;
                if t_fall < 0
                    continue;
                end
                if t_e > max(t_m_max)
                    continue;
                end

                idx = idx + 1;
                pos_drop = UAV_init(i, :) + v_vec * t_d;
                cloud_pos_exp(idx, :) = pos_drop + v_vec * t_fall - [0, 0, 0.5 * g * t_fall^2];
                cloud_t_start(idx) = t_e;
                cloud_t_end(idx) = t_e + life_cloud;
            end
        end
    end

    cloud_t_start(idx+1:end) = [];
    cloud_t_end(idx+1:end) = [];
    cloud_pos_exp(idx+1:end, :) = [];
    num_clouds = idx;

    if num_clouds == 0
        total_cover_time = 0;
        missile_cover_times = zeros(1, num_missiles);
        return;
    end

    %% 4. Single-cloud intervals first, exact multi-cloud only where needed
    missile_cover_times = zeros(1, num_missiles);
    missile_cover_intervals = cell(num_missiles, 1);

    for j = 1:num_missiles
        single_full_intervals = zeros(0, 2);
        partial_by_cloud = NaN(num_clouds, 2);

        for c = 1:num_clouds
            [partial_interval, full_interval] = find_single_cloud_intervals(j, c);

            if ~isempty(partial_interval)
                partial_by_cloud(c, :) = partial_interval;
            end

            if ~isempty(full_interval)
                single_full_intervals = [single_full_intervals; full_interval]; %#ok<AGROW>
            end
        end

        single_full_intervals = merge_intervals(single_full_intervals);

        multi_windows = find_overlap_windows(partial_by_cloud, 2);
        multi_windows = subtract_intervals(multi_windows, single_full_intervals);
        multi_intervals = evaluate_multi_windows(j, multi_windows, partial_by_cloud);

        missile_cover_intervals{j} = merge_intervals([single_full_intervals; multi_intervals]);
        missile_cover_times(j) = interval_total_length(missile_cover_intervals{j});
    end

    all_cover_intervals = zeros(0, 2);
    for j = 1:num_missiles
        all_cover_intervals = [all_cover_intervals; missile_cover_intervals{j}]; %#ok<AGROW>
    end
    total_cover_time = interval_total_length(all_cover_intervals);

    function [partial_interval, full_interval] = find_single_cloud_intervals(missile_id, cloud_id)
        partial_interval = zeros(0, 2);
        full_interval = zeros(0, 2);

        t0 = max(0, cloud_t_start(cloud_id));
        t1 = min(cloud_t_end(cloud_id), t_m_max(missile_id));
        if t1 <= t0
            return;
        end

        scan_times = make_scan_grid(t0, t1, dt_bracket);
        states = single_cloud_states(scan_times, missile_id, cloud_id);

        partial_interval = refine_interval(scan_times, states >= 1, ...
            @(t) single_cloud_state(t, missile_id, cloud_id) >= 1);
        full_interval = refine_interval(scan_times, states == 2, ...
            @(t) single_cloud_state(t, missile_id, cloud_id) == 2);
    end

    function interval = refine_interval(scan_times, flags, predicate)
        interval = zeros(0, 2);
        if ~any(flags)
            return;
        end

        first_idx = find(flags, 1, 'first');
        last_idx = find(flags, 1, 'last');

        if first_idx == 1
            left_t = scan_times(1);
        else
            left_t = bisect_first_true(scan_times(first_idx - 1), scan_times(first_idx), predicate);
        end

        if last_idx == numel(flags)
            right_t = scan_times(end);
        else
            right_t = bisect_last_true(scan_times(last_idx), scan_times(last_idx + 1), predicate);
        end

        if right_t - left_t > tol_time
            interval = [left_t, right_t];
        end
    end

    function states = single_cloud_states(times, missile_id, cloud_id)
        t = times(:);
        pos_M = M_init(missile_id, :) + (v_m * t) * M_dirs(missile_id, :);

        centers = repmat(cloud_pos_exp(cloud_id, :), numel(t), 1);
        centers(:, 3) = centers(:, 3) - v_cloud_down * (t - cloud_t_start(cloud_id));

        v_MC = centers - pos_M;
        v_MP_1 = target_pts(:, 1).' - pos_M(:, 1);
        v_MP_2 = target_pts(:, 2).' - pos_M(:, 2);
        v_MP_3 = target_pts(:, 3).' - pos_M(:, 3);

        len_MP_sq = v_MP_1.^2 + v_MP_2.^2 + v_MP_3.^2;
        len_MP_sq = max(len_MP_sq, eps);

        proj = v_MP_1 .* v_MC(:, 1) + v_MP_2 .* v_MC(:, 2) + v_MP_3 .* v_MC(:, 3);
        u = proj ./ len_MP_sq;
        u = min(1, max(0, u));

        dist_sq = (u .* v_MP_1 - v_MC(:, 1)).^2 + ...
                  (u .* v_MP_2 - v_MC(:, 2)).^2 + ...
                  (u .* v_MP_3 - v_MC(:, 3)).^2;

        blocked = dist_sq <= r_cloud_sq;
        any_blocked = any(blocked, 2);
        all_blocked = all(blocked, 2);

        states = double(any_blocked);
        states(all_blocked) = 2;
    end

    function t = bisect_first_true(lo, hi, predicate)
        while hi - lo > tol_time
            mid = 0.5 * (lo + hi);
            if predicate(mid)
                hi = mid;
            else
                lo = mid;
            end
        end
        t = hi;
    end

    function t = bisect_last_true(lo, hi, predicate)
        while hi - lo > tol_time
            mid = 0.5 * (lo + hi);
            if predicate(mid)
                lo = mid;
            else
                hi = mid;
            end
        end
        t = lo;
    end

    function state = single_cloud_state(t, missile_id, cloud_id)
        if t < cloud_t_start(cloud_id) || t > cloud_t_end(cloud_id) || t > t_m_max(missile_id)
            state = 0;
            return;
        end

        pos_M = missile_position(t, missile_id);
        center = cloud_center(t, cloud_id);
        blocked = points_blocked_by_cloud(pos_M, center);

        if all(blocked)
            state = 2;
        elseif any(blocked)
            state = 1;
        else
            state = 0;
        end
    end

    function tf = multi_smoke_full(t, missile_id, cloud_ids)
        if t > t_m_max(missile_id)
            tf = false;
            return;
        end

        pos_M = missile_position(t, missile_id);
        blocked = false(num_pts, 1);

        for n = 1:numel(cloud_ids)
            cloud_id = cloud_ids(n);
            if t < cloud_t_start(cloud_id) || t > cloud_t_end(cloud_id)
                continue;
            end

            center = cloud_center(t, cloud_id);
            blocked = blocked | points_blocked_by_cloud(pos_M, center);
            if all(blocked)
                tf = true;
                return;
            end
        end

        tf = all(blocked);
    end

    function blocked = points_blocked_by_cloud(pos_M, center)
        v_MP = bsxfun(@minus, target_pts, pos_M);
        len_MP_sq = sum(v_MP.^2, 2);
        len_MP_sq = max(len_MP_sq, eps);

        v_MC = center - pos_M;
        u = (v_MP * v_MC.') ./ len_MP_sq;
        u = min(1, max(0, u));

        diff_pts = bsxfun(@minus, bsxfun(@times, v_MP, u), v_MC);
        dist_sq = sum(diff_pts.^2, 2);

        blocked = dist_sq <= r_cloud_sq;
    end

    function pos_M = missile_position(t, missile_id)
        pos_M = M_init(missile_id, :) + (v_m * t) * M_dirs(missile_id, :);
    end

    function center = cloud_center(t, cloud_id)
        center = cloud_pos_exp(cloud_id, :);
        center(3) = center(3) - v_cloud_down * (t - cloud_t_start(cloud_id));
    end

    function windows = find_overlap_windows(intervals, min_count)
        valid = all(isfinite(intervals), 2) & intervals(:, 2) > intervals(:, 1);
        intervals = intervals(valid, :);

        if size(intervals, 1) < min_count
            windows = zeros(0, 2);
            return;
        end

        edges = unique(intervals(:));
        windows = zeros(0, 2);
        for q = 1:numel(edges)-1
            left_t = edges(q);
            right_t = edges(q + 1);
            if right_t - left_t <= tol_time
                continue;
            end

            mid_t = 0.5 * (left_t + right_t);
            overlap_count = sum(intervals(:, 1) <= mid_t & intervals(:, 2) >= mid_t);
            if overlap_count >= min_count
                windows = [windows; left_t, right_t]; %#ok<AGROW>
            end
        end

        windows = merge_intervals(windows);
    end

    function intervals = evaluate_multi_windows(missile_id, windows, partial_by_cloud)
        intervals = zeros(0, 2);
        if isempty(windows)
            return;
        end

        for w = 1:size(windows, 1)
            t0 = windows(w, 1);
            t1 = windows(w, 2);
            sample_times = make_multi_grid(t0, t1, dt_multi);
            flags = false(numel(sample_times), 1);

            for s = 1:numel(sample_times)
                t = sample_times(s);
                cloud_ids = find(partial_by_cloud(:, 1) <= t & partial_by_cloud(:, 2) >= t);
                if numel(cloud_ids) < 2
                    continue;
                end

                flags(s) = multi_smoke_full(t, missile_id, cloud_ids);
            end

            intervals = [intervals; flags_to_intervals(sample_times, flags, t1)]; %#ok<AGROW>
        end

        intervals = merge_intervals(intervals);
    end

    function intervals = flags_to_intervals(sample_times, flags, window_end)
        intervals = zeros(0, 2);
        idx_flag = 1;
        num_flags = numel(flags);

        while idx_flag <= num_flags
            if ~flags(idx_flag)
                idx_flag = idx_flag + 1;
                continue;
            end

            start_t = sample_times(idx_flag);
            while idx_flag <= num_flags && flags(idx_flag)
                idx_flag = idx_flag + 1;
            end

            last_true_idx = idx_flag - 1;
            end_t = min(sample_times(last_true_idx) + dt_multi, window_end);
            if end_t > start_t
                intervals = [intervals; start_t, end_t]; %#ok<AGROW>
            end
        end
    end

    function times = make_scan_grid(t0, t1, step)
        times = t0:step:t1;
        if isempty(times)
            times = [t0, t1];
        elseif times(end) < t1 - tol_time
            times = [times, t1];
        else
            times(end) = t1;
        end
    end

    function times = make_multi_grid(t0, t1, step)
        if t1 <= t0
            times = zeros(1, 0);
            return;
        end

        num_steps = max(1, ceil((t1 - t0) / step));
        times = t0 + (0:num_steps-1) * step;
        times = times(times < t1);
        if isempty(times)
            times = t0;
        end
    end

    function merged = merge_intervals(intervals)
        if isempty(intervals)
            merged = zeros(0, 2);
            return;
        end

        valid = all(isfinite(intervals), 2) & intervals(:, 2) > intervals(:, 1);
        intervals = intervals(valid, :);
        if isempty(intervals)
            merged = zeros(0, 2);
            return;
        end

        intervals = sortrows(intervals, 1);
        merged = intervals(1, :);
        for q = 2:size(intervals, 1)
            if intervals(q, 1) <= merged(end, 2) + tol_time
                merged(end, 2) = max(merged(end, 2), intervals(q, 2));
            else
                merged = [merged; intervals(q, :)]; %#ok<AGROW>
            end
        end
    end

    function remain = subtract_intervals(intervals, blockers)
        intervals = merge_intervals(intervals);
        blockers = merge_intervals(blockers);

        if isempty(intervals) || isempty(blockers)
            remain = intervals;
            return;
        end

        remain = zeros(0, 2);
        for q = 1:size(intervals, 1)
            current_start = intervals(q, 1);
            current_end = intervals(q, 2);

            for b = 1:size(blockers, 1)
                if blockers(b, 2) <= current_start + tol_time
                    continue;
                end
                if blockers(b, 1) >= current_end - tol_time
                    break;
                end

                if blockers(b, 1) > current_start + tol_time
                    remain = [remain; current_start, min(blockers(b, 1), current_end)]; %#ok<AGROW>
                end
                current_start = max(current_start, blockers(b, 2));

                if current_start >= current_end - tol_time
                    break;
                end
            end

            if current_start < current_end - tol_time
                remain = [remain; current_start, current_end]; %#ok<AGROW>
            end
        end

        remain = merge_intervals(remain);
    end

    function total_len = interval_total_length(intervals)
        intervals = merge_intervals(intervals);
        if isempty(intervals)
            total_len = 0;
        else
            total_len = sum(intervals(:, 2) - intervals(:, 1));
        end
    end
end
