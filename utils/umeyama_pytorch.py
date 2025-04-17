import torch

def estimateSimilarityUmeyama(source: torch.Tensor, target: torch.Tensor):
    """
    PyTorch version of the Umeyama algorithm for estimating similarity transformation.
    
    Args:
        source: A tensor of shape (4, N) representing the homogeneous source points.
        target: A tensor of shape (4, N) representing the homogeneous target points.
        
    Returns:
        scale: Scaling factor.
        rotation: 3x3 rotation matrix.
        translation: 3x1 translation vector.
        transformation: 4x4 similarity transformation matrix.
    """
    # Compute centroids
    source_centroid = torch.mean(source[:3, :], dim=1)
    target_centroid = torch.mean(target[:3, :], dim=1)
    
    # Centralize points
    centered_source = source[:3, :] - source_centroid.unsqueeze(1)
    centered_target = target[:3, :] - target_centroid.unsqueeze(1)
    
    # Covariance matrix
    cov_matrix = torch.matmul(centered_target, centered_source.transpose(0, 1)) / source.shape[1]

    # SVD decomposition
    U, D, Vh = torch.linalg.svd(cov_matrix, full_matrices=True)
    
    # Handle reflection case
    d = (torch.det(U) * torch.det(Vh)) < 0.0
    if d:
        D[-1] = -D[-1]
        U[:, -1] = -U[:, -1]
    
    # Rotation matrix
    rotation = torch.matmul(U, Vh)

    # Scale factor
    # if source[:3, :].size(1) < 2:  # 每行至少需要两个样本点
    #     print("source[:3, :] 数据不足，跳过当前操作。")
    #     import pdb
    #     pdb.set_trace()
    # else:
    #     varP = torch.var(source[:3, :], dim=1).sum()
    varP = torch.var(source[:3, :], dim=1).sum()
    scale = torch.sum(D) / varP
    
    # Translation vector
    translation = target_centroid - scale * torch.matmul(rotation, source_centroid)
    
    # Transformation matrix
    out_transform = torch.eye(4, device=source.device)
    out_transform[:3, :3] = scale * rotation
    out_transform[:3, 3] = translation

    return scale, rotation, translation, out_transform


def estimateSimilarityTransform(source: torch.Tensor, target: torch.Tensor, verbose=False):
    """
    PyTorch version of RANSAC-based similarity transform estimation.
    
    Args:
        source: A tensor of shape (N, 3) representing the source points.
        target: A tensor of shape (N, 3) representing the target points.
        verbose: Whether to print detailed logs during RANSAC.
        
    Returns:
        scale: Scaling factor.
        rotation: 3x3 rotation matrix.
        translation: 3x1 translation vector.
        transformation: 4x4 similarity transformation matrix.
    """
    assert source.shape[0] == target.shape[0], "Source and Target must have the same number of points."

    # Add homogeneous coordinate
    source_hom = torch.cat([source, torch.ones(source.shape[0], 1, device=source.device)], dim=1).transpose(0, 1)
    target_hom = torch.cat([target, torch.ones(target.shape[0], 1, device=target.device)], dim=1).transpose(0, 1)

    # Compute centroids and centralized coordinates
    source_centroid = torch.mean(source_hom[:3, :], dim=1)
    n_points = source_hom.shape[1]
    centered_source = source_hom[:3, :] - source_centroid.unsqueeze(1)
    source_diameter = 2 * torch.max(torch.norm(centered_source, dim=0))

    # RANSAC parameters
    inlier_t = source_diameter / 10  # 0.1 of source diameter
    max_iter = 128
    confidence = 0.99

    if verbose:
        print(f'Inlier threshold: {inlier_t.item()}')
        print(f'Max number of iterations: {max_iter}')

    best_inlier_ratio = 0
    best_inlier_idx = torch.arange(n_points, device=source.device)

    for i in range(max_iter):
        # Randomly select 5 corresponding points
        rand_idx = torch.randint(0, n_points, (10,), device=source.device)
        scale, _, _, out_transform = estimateSimilarityUmeyama(source_hom[:, rand_idx], target_hom[:, rand_idx])
        
        pass_threshold = scale * inlier_t
        diff = target_hom - torch.matmul(out_transform, source_hom)
        residual_vec = torch.norm(diff[:3, :], dim=0)
        
        inlier_idx = torch.where(residual_vec < pass_threshold)[0]
        n_inliers = inlier_idx.shape[0]
        inlier_ratio = n_inliers / n_points

        # Update best inlier ratio
        if inlier_ratio > best_inlier_ratio:
            best_inlier_ratio = inlier_ratio
            best_inlier_idx = inlier_idx
        
        if verbose:
            print(f'Iteration: {i}')
            print(f'Inlier ratio: {best_inlier_ratio}')

        # Early termination based on RANSAC confidence
        if (1 - (1 - best_inlier_ratio ** 10) ** i) > confidence:
            break

    if best_inlier_ratio < 0.1:
        #print('[ WARN ] - Small BestInlierRatio:', best_inlier_ratio)
        return None, None, None, None, None

    source_inliers_hom = source_hom[:, best_inlier_idx]
    target_inliers_hom = target_hom[:, best_inlier_idx]
    scale, rotation, translation, out_transform = estimateSimilarityUmeyama(source_inliers_hom, target_inliers_hom)
    # import pdb
    # pdb.set_trace()
    if verbose:
        print('BestInlierRatio:', best_inlier_ratio.item())
        print('Rotation:\n', rotation)
        print('Translation:\n', translation)
        print('Scale:', scale)

    return scale, rotation, translation, out_transform,best_inlier_idx