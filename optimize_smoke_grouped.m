% OPTIMIZE_SMOKE_GROUPED
% Greedy grouped optimizer for the 40-element vector used by
% run_smoke_simulation(x).
%
% x layout:
%   1:5   UAV speed
%   6:10  UAV heading phi
%   11:15 first bomb drop time
%   16:20 extra gap from bomb 1 to bomb 2; actual gap is 1 + extra
%   21:25 extra gap from bomb 2 to bomb 3; actual gap is 1 + extra
%   26:40 fuse delay matrix, row-wise by UAV:
%         [uav1 bomb1 bomb2 bomb3, uav2 bomb1 bomb2 bomb3, ...]
%
% Optimization order:
%   For each UAV i:
%     1) optimize [v_i, phi_i, t_first_i, fuse_i1]
%     2) fix v_i and phi_i, optimize [dt12_extra_i, fuse_i2]
%     3) fix v_i and phi_i, optimize [dt23_extra_i, fuse_i3]
%   Inside each small group, optimize M1, M2 and M3 separately.  The
%   acceptance rule for a target missile only looks at that missile's own
%   cover-time increase, not at the already accumulated total score.
%
% The algorithm is intentionally toolbox-free.  Each small group uses
% deterministic physics-biased seeds, random coarse search, then coordinate
% pattern search.  It is not a global optimizer, but it avoids the original
% 40-D all-at-once search where most samples return zero.

clear;
clc;
rng(20260705, 'twister');

cfg = struct();
cfg.numUavs = 5;
cfg.numBombs = 3;
cfg.numMissiles = 3;
cfg.targetOrder = [1, 2, 3];
cfg.disabledDelay = 9999;

cfg.vBounds = [70, 140];
cfg.phiBounds = [-pi, pi];
cfg.firstDropBounds = [0, 55];
cfg.gapExtraBounds = [0, 20];
cfg.fuseBounds = [0, 18];

% The optimization score is the sum of the three per-missile union times.
% The model's first output total_time is still recorded as union_total for
% reference, but it is no longer used as the optimization objective.
cfg.objectiveMode = 'sum_missiles';

cfg.firstRandomSamples = 600;
cfg.laterRandomSamples = 250;
cfg.maxLocalEvals = 300;
cfg.localShrink = 0.5;
cfg.scoreTol = 1e-9;

cfg.firstInitialStep = [12, pi / 4, 6, 3];
cfg.firstMinStep = [0.25, pi / 360, 0.05, 0.05];
cfg.laterInitialStep = [5, 3];
cfg.laterMinStep = [0.05, 0.05];

cfg.saveFile = 'best_grouped_solution.mat';

xBest = make_initial_x(cfg);
[bestObj, bestTotalTime, bestMissileTimes] = evaluate_x(xBest, cfg);

history = repmat(struct( ...
    'stage', '', ...
    'uav', 0, ...
    'bomb', 0, ...
    'indices', [], ...
    'objective', 0, ...
    'total_time', 0, ...
    'missile_times', zeros(1, 3), ...
    'target_gains', zeros(1, 3), ...
    'sum_gain', 0, ...
    'evals', 0, ...
    'x', zeros(40, 1)), 0, 1);

fprintf('Initial score(sum missiles): %.6f, union_total: %.6f\n', bestObj, bestTotalTime);

for uavId = 1:cfg.numUavs
    % First bomb: train UAV speed, heading, first drop time, first fuse.
    groupIdx = [uavId, 5 + uavId, 10 + uavId, delay_index(uavId, 1)];
    lb = [cfg.vBounds(1), cfg.phiBounds(1), cfg.firstDropBounds(1), cfg.fuseBounds(1)];
    ub = [cfg.vBounds(2), cfg.phiBounds(2), cfg.firstDropBounds(2), cfg.fuseBounds(2)];
    seedRows = first_bomb_seed_rows(uavId, cfg);

    [xBest, bestObj, bestTotalTime, bestMissileTimes, info] = optimize_group( ...
        xBest, groupIdx, lb, ub, seedRows, cfg.firstRandomSamples, ...
        cfg.firstInitialStep, cfg.firstMinStep, cfg);

    history(end + 1) = make_history(sprintf('uav%d_bomb1', uavId), uavId, 1, ...
        groupIdx, bestObj, bestTotalTime, bestMissileTimes, ...
        info.target_gains, info.sum_gain, info.evals, xBest); %#ok<SAGROW>
    print_stage(history(end));

    % Second bomb: keep UAV speed and heading fixed, train gap and fuse.
    groupIdx = [15 + uavId, delay_index(uavId, 2)];
    lb = [cfg.gapExtraBounds(1), cfg.fuseBounds(1)];
    ub = [cfg.gapExtraBounds(2), cfg.fuseBounds(2)];
    seedRows = later_bomb_seed_rows(cfg);

    [xBest, bestObj, bestTotalTime, bestMissileTimes, info] = optimize_group( ...
        xBest, groupIdx, lb, ub, seedRows, cfg.laterRandomSamples, ...
        cfg.laterInitialStep, cfg.laterMinStep, cfg);

    history(end + 1) = make_history(sprintf('uav%d_bomb2', uavId), uavId, 2, ...
        groupIdx, bestObj, bestTotalTime, bestMissileTimes, ...
        info.target_gains, info.sum_gain, info.evals, xBest); %#ok<SAGROW>
    print_stage(history(end));

    % Third bomb: keep UAV speed and heading fixed, train gap and fuse.
    groupIdx = [20 + uavId, delay_index(uavId, 3)];
    lb = [cfg.gapExtraBounds(1), cfg.fuseBounds(1)];
    ub = [cfg.gapExtraBounds(2), cfg.fuseBounds(2)];
    seedRows = later_bomb_seed_rows(cfg);

    [xBest, bestObj, bestTotalTime, bestMissileTimes, info] = optimize_group( ...
        xBest, groupIdx, lb, ub, seedRows, cfg.laterRandomSamples, ...
        cfg.laterInitialStep, cfg.laterMinStep, cfg);

    history(end + 1) = make_history(sprintf('uav%d_bomb3', uavId), uavId, 3, ...
        groupIdx, bestObj, bestTotalTime, bestMissileTimes, ...
        info.target_gains, info.sum_gain, info.evals, xBest); %#ok<SAGROW>
    print_stage(history(end));
end

best_x = xBest; %#ok<NASGU>
best_objective = bestObj; %#ok<NASGU>
best_total_time = bestTotalTime; %#ok<NASGU>
best_missile_times = bestMissileTimes; %#ok<NASGU>

save(cfg.saveFile, 'best_x', 'best_objective', 'best_total_time', ...
    'best_missile_times', 'history', 'cfg');

fprintf('\nDone.\n');
fprintf('Best score(sum missiles): %.6f\n', best_objective);
fprintf('Best union_total from model: %.6f\n', best_total_time);
fprintf('Best missile_times: [%.6f %.6f %.6f]\n', best_missile_times);
fprintf('Best x 40x1 vector:\n');
fprintf('best_x = [\n');
for idx = 1:numel(best_x)
    if idx < numel(best_x)
        fprintf('    %.12g;\n', best_x(idx));
    else
        fprintf('    %.12g\n', best_x(idx));
    end
end
fprintf('];\n');
fprintf('Saved: %s\n', cfg.saveFile);

function x = make_initial_x(cfg)
    x = zeros(40, 1);
    x(1:5) = 100;
    x(6:10) = 0;
    x(11:15) = 0;
    x(16:20) = 0;
    x(21:25) = 0;
    x(26:40) = cfg.disabledDelay;
end

function idx = delay_index(uavId, bombId)
    idx = 25 + (uavId - 1) * 3 + bombId;
end

function rows = first_bomb_seed_rows(uavId, cfg)
    uavInit = [
        17800, 0, 1800;
        12000, 1400, 1400;
        6000, -3000, 700;
        11000, 2000, 1800;
        13000, -2000, 1300
    ];
    missileInit = [
        20000, 0, 2000;
        19000, 600, 2100;
        18000, -600, 1900
    ];
    targetXY = [0, 200];
    targetPoint = [0, 200, 5];
    missileSpeed = 300;

    basePhi = atan2(targetXY(2) - uavInit(uavId, 2), ...
        targetXY(1) - uavInit(uavId, 1));

    speedSeeds = [120, 100, 140, 80];
    phiSeeds = wrap_pi(basePhi + [0, -0.2, 0.2, -0.5, 0.5]);
    dropSeeds = [0, 1.5, 4, 8, 15, 30, 50];
    fuseSeeds = [1, 3.6, 6, 10, 15];

    rows = zeros(numel(speedSeeds) * numel(phiSeeds) * ...
        numel(dropSeeds) * numel(fuseSeeds), 4);
    rowId = 0;

    for v = speedSeeds
        for phi = phiSeeds
            for dropTime = dropSeeds
                for fuseDelay = fuseSeeds
                    rowId = rowId + 1;
                    rows(rowId, :) = [v, phi, dropTime, fuseDelay];
                end
            end
        end
    end

    guidedRows = zeros(0, 4);
    guideTimes = [8, 12, 16, 20, 25, 30, 35, 40, 45, 50, 55, 60];
    lineFractions = [0.15, 0.25, 0.35, 0.50, 0.70];
    guideFuseSeeds = [1, 3.6, 6, 10, 15];

    for missileId = 1:size(missileInit, 1)
        missileDir = -missileInit(missileId, :) / norm(missileInit(missileId, :));
        missileMaxTime = norm(missileInit(missileId, :)) / missileSpeed;

        for explodeTime = guideTimes
            if explodeTime >= missileMaxTime
                continue;
            end

            missilePos = missileInit(missileId, :) + missileSpeed * explodeTime * missileDir;

            for frac = lineFractions
                desiredPos = missilePos + frac * (targetPoint - missilePos);
                vecXY = desiredPos(1:2) - uavInit(uavId, 1:2);
                distXY = norm(vecXY);
                if distXY <= 1e-9
                    continue;
                end

                vGuide = distXY / explodeTime;
                if vGuide < cfg.vBounds(1) || vGuide > cfg.vBounds(2)
                    continue;
                end

                phiGuide = atan2(vecXY(2), vecXY(1));
                for fuseDelay = guideFuseSeeds
                    dropTime = explodeTime - fuseDelay;
                    if dropTime < cfg.firstDropBounds(1) || dropTime > cfg.firstDropBounds(2)
                        continue;
                    end
                    guidedRows(end + 1, :) = [vGuide, phiGuide, dropTime, fuseDelay]; %#ok<AGROW>
                end
            end
        end
    end

    rows = [rows; guidedRows]; %#ok<AGROW>

    rows(:, 1) = clamp(rows(:, 1), cfg.vBounds);
    rows(:, 2) = clamp(wrap_pi(rows(:, 2)), cfg.phiBounds);
    rows(:, 3) = clamp(rows(:, 3), cfg.firstDropBounds);
    rows(:, 4) = clamp(rows(:, 4), cfg.fuseBounds);
end

function rows = later_bomb_seed_rows(cfg)
    gapSeeds = [0, 1, 3, 6, 10, 15, 20];
    fuseSeeds = [1, 3.6, 6, 10, 15];

    rows = zeros(numel(gapSeeds) * numel(fuseSeeds), 2);
    rowId = 0;

    for gapExtra = gapSeeds
        for fuseDelay = fuseSeeds
            rowId = rowId + 1;
            rows(rowId, :) = [gapExtra, fuseDelay];
        end
    end

    rows(:, 1) = clamp(rows(:, 1), cfg.gapExtraBounds);
    rows(:, 2) = clamp(rows(:, 2), cfg.fuseBounds);
end

function [xBest, bestObj, bestTotalTime, bestMissileTimes, info] = optimize_group( ...
    xBase, groupIdx, lb, ub, seedRows, randomSamples, initialStep, minStep, cfg)

    xBest = xBase;
    [startObj, startTotalTime, startMissileTimes] = evaluate_x(xBest, cfg);
    bestObj = startObj;
    bestTotalTime = startTotalTime;
    bestMissileTimes = startMissileTimes;

    evals = 1;
    candidateRows = build_candidate_rows(xBase, groupIdx, lb, ub, seedRows, randomSamples);

    for targetId = cfg.targetOrder
        [xCand, candTotalTime, candMissileTimes, targetInfo] = optimize_group_for_target( ...
            xBase, groupIdx, lb, ub, candidateRows, initialStep, minStep, targetId, cfg);

        evals = evals + targetInfo.evals;
        targetGain = candMissileTimes(targetId) - startMissileTimes(targetId);
        candObj = sum(candMissileTimes);

        if targetGain > cfg.scoreTol && candObj > bestObj + cfg.scoreTol
            xBest = xCand;
            bestObj = candObj;
            bestTotalTime = candTotalTime;
            bestMissileTimes = candMissileTimes;
        end
    end

    [bestObj, bestTotalTime, bestMissileTimes] = evaluate_x(xBest, cfg);
    evals = evals + 1;
    targetGains = bestMissileTimes - startMissileTimes;

    info = struct( ...
        'evals', evals, ...
        'target_gains', targetGains, ...
        'sum_gain', bestObj - startObj, ...
        'start_missile_times', startMissileTimes);
end

function [xBest, bestTotalTime, bestMissileTimes, info] = optimize_group_for_target( ...
    xBase, groupIdx, lb, ub, candidateRows, initialStep, minStep, targetId, cfg)

    xBest = xBase;
    [~, bestTotalTime, bestMissileTimes] = evaluate_x(xBest, cfg);
    bestTargetTime = bestMissileTimes(targetId);
    evals = 1;

    for rowId = 1:size(candidateRows, 1)
        xCand = set_group_values(xBase, groupIdx, candidateRows(rowId, :), lb, ub);
        [~, totalTime, missileTimes] = evaluate_x(xCand, cfg);
        evals = evals + 1;

        if missileTimes(targetId) > bestTargetTime + cfg.scoreTol
            xBest = xCand;
            bestTargetTime = missileTimes(targetId);
            bestTotalTime = totalTime;
            bestMissileTimes = missileTimes;
        end
    end

    step = initialStep(:).';
    minStep = minStep(:).';
    localEvals = 0;

    while any(step > minStep) && localEvals < cfg.maxLocalEvals
        improved = false;

        for dimId = 1:numel(groupIdx)
            for direction = [-1, 1]
                row = xBest(groupIdx).';
                row(dimId) = row(dimId) + direction * step(dimId);
                xCand = set_group_values(xBest, groupIdx, row, lb, ub);

                [~, totalTime, missileTimes] = evaluate_x(xCand, cfg);
                evals = evals + 1;
                localEvals = localEvals + 1;

                if missileTimes(targetId) > bestTargetTime + cfg.scoreTol
                    xBest = xCand;
                    bestTargetTime = missileTimes(targetId);
                    bestTotalTime = totalTime;
                    bestMissileTimes = missileTimes;
                    improved = true;
                end

                if localEvals >= cfg.maxLocalEvals
                    break;
                end
            end

            if localEvals >= cfg.maxLocalEvals
                break;
            end
        end

        if ~improved
            step = step * cfg.localShrink;
        end
    end

    info = struct('evals', evals);
end

function candidateRows = build_candidate_rows(xBase, groupIdx, lb, ub, seedRows, randomSamples)
    currentRow = xBase(groupIdx).';
    if all(isfinite(currentRow))
        candidateRows = [currentRow; seedRows]; %#ok<AGROW>
    else
        candidateRows = seedRows;
    end

    if randomSamples > 0
        randomRows = lb + rand(randomSamples, numel(groupIdx)) .* (ub - lb);
        candidateRows = [candidateRows; randomRows]; %#ok<AGROW>
    end
end

function x = set_group_values(x, groupIdx, row, lb, ub)
    row = row(:).';
    for k = 1:numel(groupIdx)
        value = row(k);
        if groupIdx(k) >= 6 && groupIdx(k) <= 10
            value = wrap_pi(value);
        end
        x(groupIdx(k)) = min(max(value, lb(k)), ub(k));
    end
end

function [objective, totalTime, missileTimes] = evaluate_x(x, cfg)
    try
        [totalTime, missileTimes] = run_smoke_simulation(x);
        if isempty(missileTimes)
            missileTimes = zeros(1, 3);
        end
        missileTimes = missileTimes(:).';

        if strcmp(cfg.objectiveMode, 'sum_missiles')
            objective = sum(missileTimes);
        else
            objective = totalTime;
        end

        if ~isfinite(objective)
            objective = -Inf;
        end
    catch
        objective = -Inf;
        totalTime = 0;
        missileTimes = zeros(1, 3);
    end
end

function entry = make_history(stageName, uavId, bombId, groupIdx, objective, ...
    totalTime, missileTimes, targetGains, sumGain, evals, x)
    entry = struct( ...
        'stage', stageName, ...
        'uav', uavId, ...
        'bomb', bombId, ...
        'indices', groupIdx, ...
        'objective', objective, ...
        'total_time', totalTime, ...
        'missile_times', missileTimes, ...
        'target_gains', targetGains, ...
        'sum_gain', sumGain, ...
        'evals', evals, ...
        'x', x);
end

function print_stage(entry)
    fprintf(['%s done | score %.6f | union_total %.6f | missiles [%.6f %.6f %.6f] ', ...
        '| target_gains [%.6f %.6f %.6f] | sum_gain %.6f | evals %d\n'], ...
        entry.stage, entry.objective, entry.total_time, entry.missile_times, ...
        entry.target_gains, entry.sum_gain, entry.evals);
end

function y = clamp(x, bounds)
    y = min(max(x, bounds(1)), bounds(2));
end

function angle = wrap_pi(angle)
    angle = mod(angle + pi, 2 * pi) - pi;
end
