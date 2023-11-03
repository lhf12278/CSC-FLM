# t_feat, t_score = model(t_img, modal=1, cam_label=t_target_cam)  # t_feat list 4个16×768的tensor
# # 源域的pid继续约束（leader light),仅对目标域的global feature进行聚类，利用聚类生成的伪标签进行loss约束
# t_global_feat = t_feat[0]
# dict_f, cls_score = coarse_classi(t_global_feat.detach())
# # 对t_global_feat聚类
# # cf = torch.stack(list(dict_f.values()))
# if (cfg.LAMBDA_VALUE > 0):
#     t_feat, t_score = model(img, modal=1, cam_label=target_cam)
#     dict_f = t_feat[0]
#     # cf_s = torch.stack(list(dict_f.values()))
#     rerank_dist = compute_jaccard_dist(dict_f, lambda_value=cfg.lambda_value,
#                                        source_features=dict_f,
#                                        use_gpu=cfg.RR_GPU).detach().numpy()
# else:
#     t_feat, t_score = model(img, modal=1, cam_label=target_cam)
#     dict_f = t_feat[0]
#     rerank_dist = compute_jaccard_dist(dict_f, use_gpu=cfg.RR_GPU).detach().numpy()
#
# # DBSCAN1 cluster
# tri_mat = np.triu(rerank_dist, 1)  # tri_mat.dim=2
# tri_mat = tri_mat[np.nonzero(tri_mat)]  # tri_mat.dim=1
# tri_mat = np.sort(tri_mat, axis=None)
# rho = 1.6e-3
# top_num = np.round(rho * tri_mat.size).astype(int)
# eps = tri_mat[:top_num].mean()
# print('eps for target cluster: {:.3f}'.format(eps))
# cluster = DBSCAN1(eps=0.5, min_samples=4, metric='euclidean', n_jobs=-1)
#
# print('Clustering and labeling...')
# pseudo_labels_tar = cluster.fit_predict(rerank_dist)
# # num_ids_tar = len(set(pseudo_labels_tar)) - (1 if -1 in pseudo_labels_tar else 0)
# # cfg.num_clusters_tar = num_ids_tar
# # print('\n Clustered into {} classes \n'.format(cfg.num_clusters_tar))
# pseudo_labels_tar = torch.from_numpy(pseudo_labels_tar)
# # pseudo_labels_tar = pseudo_labels_tar.tolist()
# t_ide_loss = ide_creiteron(cls_score, pseudo_labels_tar)
# # triplet = TripletLoss()
# # print(dict_f.is_cuda, pseudo_labels_tar.is_cuda) #True False
# # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# # pseudo_labels_tar.to(device)
# # dict_f.to(device)
# # t_triplet_loss = triplet(dict_f, pseudo_labels_tar)
#
# f_out_t, p_out_t = self.model(inputs)
# p_out_t = p_out_t[:, :self.num_cluster]
#
# loss_ce = criterion_ce(p_out_t, targets)
# loss_tri = criterion_tri(f_out_t, f_out_t, targets)
# loss = loss_ce + loss_tri
# t_loss = t_ide_loss
#
# # t_loss = loss_fn(t_score, t_feat, t_target, t_target_cam)
# # scaler.scale(t_loss).backward(retain_graph=True)
# # t_loss.requires_grad_(True)  # 加入此句就行了
# scaler.scale(t_loss).backward()
# # scaler.step(optimizer) #更新粗粒度和细粒度的分类器时，是否需要更新feature extractor? (optimizer)
# scaler.step(coarse_classi_optimizer)
# # scaler.step(refine_classi_optimizer)
# scaler.update()
