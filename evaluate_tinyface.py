'''
Evaluation on TinyFace

'''



import torch
import numpy as np
from tqdm import tqdm
import data_utils
import argparse 
import pandas as pd
import sys, os
from models import get_model
from network_inf import builder_inf
import net
from models.resnet import *
import scipy.io as sio
import scipy

def get_all_files(root, extension_list=['.jpg', '.png', '.jpeg']):

    all_files = list()
    for (dirpath, dirnames, filenames) in os.walk(root):
        all_files += [os.path.join(dirpath, file) for file in filenames]
    if extension_list is None:
        return all_files
    all_files = list(filter(lambda x: os.path.splitext(x)[1] in extension_list, all_files))
    return all_files


class TinyFaceTest:
    def __init__(self, tinyface_root='/data/data/faces/tinyface', alignment_dir_name='aligned_pad_0.1_pad_high'):

        self.tinyface_root = tinyface_root
        # as defined by tinyface protocol
        self.gallery_dict = scipy.io.loadmat(os.path.join(tinyface_root, 'tinyface/Testing_Set/gallery_match_img_ID_pairs.mat'))
        self.probe_dict = scipy.io.loadmat(os.path.join(tinyface_root, 'tinyface/Testing_Set/probe_img_ID_pairs.mat'))
        self.proto_gal_paths = [os.path.join(tinyface_root, alignment_dir_name, 'Gallery_Match', p[0].item()) for p in self.gallery_dict['gallery_set']]
        self.proto_prob_paths = [os.path.join(tinyface_root, alignment_dir_name, 'Probe', p[0].item()) for p in self.probe_dict['probe_set']]
        self.proto_distractor_paths = get_all_files(os.path.join(tinyface_root, alignment_dir_name, 'Gallery_Distractor'))

        self.image_paths = get_all_files(os.path.join(tinyface_root, alignment_dir_name))
        self.image_paths = np.array(self.image_paths).astype(object).flatten()

        self.probe_paths = get_all_files(os.path.join(tinyface_root, 'tinyface/Testing_Set/Probe'))
        self.probe_paths = np.array(self.probe_paths).astype(object).flatten()

        self.gallery_paths = get_all_files(os.path.join(tinyface_root, 'tinyface/Testing_Set/Gallery_Match'))
        self.gallery_paths = np.array(self.gallery_paths).astype(object).flatten()

        self.distractor_paths = get_all_files(os.path.join(tinyface_root, 'tinyface/Testing_Set/Gallery_Distractor'))
        self.distractor_paths = np.array(self.distractor_paths).astype(object).flatten()

        self.init_proto(self.probe_paths, self.gallery_paths, self.distractor_paths)

        self.total_subjects, self.total_images = self.count_subjects_and_images()

        print("Total subjects: ", self.total_subjects)
        print("Total images: ", self.total_images)
        print("labelprobe: ", len(self.labels_probe))
        print("labelmatch: ", len(self.labels_match))
        print("labeldis: ", len(self.labels_distractor))
        print("labelgal: ", len(self.labels_gallery))
        print("Number of probe images: ", len(self.probe_paths))
        print("Number of gallery images: ", len(self.gallery_paths))

    def count_subjects_and_images(self):
        # Combine labels from probe and match (excluding distractors)
        all_labels = np.concatenate([self.labels_probe, self.labels_match])

        # Count unique subjects
        unique_labels = np.unique(all_labels[all_labels != -100])
        total_subjects = len(unique_labels)

        # Count total number of images
        total_images = len(self.image_paths)

        return total_subjects, total_images
    
    def get_key(self, image_path):
        return os.path.splitext(os.path.basename(image_path))[0]

    def get_label(self, image_path):
        return int(os.path.basename(image_path).split('_')[0])

    def init_proto(self, probe_paths, match_paths, distractor_paths):
        index_dict = {}
        for i, image_path in enumerate(self.image_paths):
            index_dict[self.get_key(image_path)] = i

        self.indices_probe = np.array([index_dict[self.get_key(img)] for img in probe_paths])
        self.indices_match = np.array([index_dict[self.get_key(img)] for img in match_paths])
        self.indices_distractor = np.array([index_dict[self.get_key(img)] for img in distractor_paths])

        self.labels_probe = np.array([self.get_label(img) for img in probe_paths])
        self.labels_match = np.array([self.get_label(img) for img in match_paths])
        self.labels_distractor = np.array([-100 for img in distractor_paths])

        self.indices_gallery = np.concatenate([self.indices_match, self.indices_distractor])
        self.labels_gallery = np.concatenate([self.labels_match, self.labels_distractor])

        print("labelprobe: ", len(self.labels_probe))
        print("labelmatch: ", len(self.labels_match))
        print("labeldis: ", len(self.labels_distractor))
        print("labelgal: ", len(self.labels_gallery))
        print(len(self.probe_paths))
        print(len(self.gallery_paths))



    def test_identification(self, features, ranks=[1,5,20]):
        feat_probe = features[self.indices_probe]
        feat_gallery = features[self.indices_gallery]
        compare_func = inner_product
        score_mat = compare_func(feat_probe, feat_gallery)

        label_mat = self.labels_probe[:,None] == self.labels_gallery[None,:]

        num_positive_pairs = np.sum(label_mat)
        num_negative_pairs = label_mat.size - num_positive_pairs
        print(label_mat.size)
        print(num_positive_pairs)
        print(num_negative_pairs)

        results, _, __ = DIR_FAR(score_mat, label_mat, ranks)

        return results

def inner_product(x1, x2):
    x1, x2 = np.array(x1), np.array(x2)
    if x1.ndim == 3:
        raise ValueError('why?')
        x1, x2 = x1[:,:,0], x2[:,:,0]
    return np.dot(x1, x2.T)



def DIR_FAR(score_mat, label_mat, ranks=[1], FARs=[1.0], get_false_indices=False):
    '''
    Code borrowed from https://github.com/seasonSH/Probabilistic-Face-Embeddings

    Closed/Open-set Identification.
        A general case of Cummulative Match Characteristic (CMC)
        where thresholding is allowed for open-set identification.
    args:
        score_mat:            a P x G matrix, P is number of probes, G is size of gallery
        label_mat:            a P x G matrix, bool
        ranks:                a list of integers
        FARs:                 false alarm rates, if 1.0, closed-set identification (CMC)
        get_false_indices:    not implemented yet
    return:
        DIRs:                 an F x R matrix, F is the number of FARs, R is the number of ranks,
                              flatten into a vector if F=1 or R=1.
        FARs:                 an vector of length = F.
        thredholds:           an vector of length = F.
    '''
    assert score_mat.shape==label_mat.shape
    # assert np.all(label_mat.astype(np.float32).sum(axis=1) <=1 )
    # Split the matrix for match probes and non-match probes
    # subfix _m: match, _nm: non-match
    # For closed set, we only use the match probes
    match_indices = label_mat.astype(bool).any(axis=1)
    score_mat_m = score_mat[match_indices,:]
    label_mat_m = label_mat[match_indices,:]
    score_mat_nm = score_mat[np.logical_not(match_indices),:]
    label_mat_nm = label_mat[np.logical_not(match_indices),:]

    print('mate probes: %d, non mate probes: %d' % (score_mat_m.shape[0], score_mat_nm.shape[0]))

    # Find the thresholds for different FARs
    max_score_nm = np.max(score_mat_nm, axis=1)
    label_temp = np.zeros(max_score_nm.shape, dtype=bool)
    if len(FARs) == 1 and FARs[0] >= 1.0:
        # If only testing closed-set identification, use the minimum score as threshold
        # in case there is no non-mate probes
        thresholds = [np.min(score_mat) - 1e-10]
        openset = False
    else:
        # If there is open-set identification, find the thresholds by FARs.
        assert score_mat_nm.shape[0] > 0, "For open-set identification (FAR<1.0), there should be at least one non-mate probe!"
        thresholds = find_thresholds_by_FAR(max_score_nm, label_temp, FARs=FARs)
        openset = True

    # Sort the labels row by row according to scores
    sort_idx_mat_m = np.argsort(score_mat_m, axis=1)
    sorted_label_mat_m = np.ndarray(label_mat_m.shape, dtype=bool)
    for row in range(label_mat_m.shape[0]):
        sort_idx = (sort_idx_mat_m[row, :])[::-1]
        sorted_label_mat_m[row,:] = label_mat_m[row, sort_idx]

    # Calculate DIRs for different FARs and ranks
    if openset:
        gt_score_m = score_mat_m[label_mat_m]
        assert gt_score_m.size == score_mat_m.shape[0]

    DIRs = np.zeros([len(FARs), len(ranks)], dtype=np.float32)
    FARs = np.zeros([len(FARs)], dtype=np.float32)
    if get_false_indices:
        false_retrieval = np.zeros([len(FARs), len(ranks), score_mat_m.shape[0]], dtype=bool)
        false_reject = np.zeros([len(FARs), len(ranks), score_mat_m.shape[0]], dtype=bool)
        false_accept = np.zeros([len(FARs), len(ranks), score_mat_nm.shape[0]], dtype=bool)
    for i, threshold in enumerate(thresholds):
        for j, rank  in enumerate(ranks):
            success_retrieval = sorted_label_mat_m[:,0:rank].any(axis=1)
            if openset:
                success_threshold = gt_score_m >= threshold
                DIRs[i,j] = (success_threshold & success_retrieval).astype(np.float32).mean()
            else:
                DIRs[i,j] = success_retrieval.astype(np.float32).mean()
            if get_false_indices:
                false_retrieval[i,j] = ~success_retrieval
                false_accept[i,j] = score_mat_nm.max(1) >= threshold
                if openset:
                    false_reject[i,j] = ~success_threshold
        if score_mat_nm.shape[0] > 0:
            FARs[i] = (max_score_nm >= threshold).astype(np.float32).mean()

    if DIRs.shape[0] == 1 or DIRs.shape[1] == 1:
        DIRs = DIRs.flatten()

    if get_false_indices:
        return DIRs, FARs, thresholds, match_indices, false_retrieval, false_reject, false_accept, sort_idx_mat_m
    else:
        return DIRs, FARs, thresholds


# Find thresholds given FARs
# but the real FARs using these thresholds could be different
# the exact FARs need to recomputed using calcROC
def find_thresholds_by_FAR(score_vec, label_vec, FARs=None, epsilon=1e-5):
    #     Code borrowed from https://github.com/seasonSH/Probabilistic-Face-Embeddings

    assert len(score_vec.shape)==1
    assert score_vec.shape == label_vec.shape
    assert label_vec.dtype == bool
    score_neg = score_vec[~label_vec]
    score_neg[::-1].sort()
    # score_neg = np.sort(score_neg)[::-1] # score from high to low
    num_neg = len(score_neg)

    assert num_neg >= 1

    if FARs is None:
        thresholds = np.unique(score_neg)
        thresholds = np.insert(thresholds, 0, thresholds[0]+epsilon)
        thresholds = np.insert(thresholds, thresholds.size, thresholds[-1]-epsilon)
    else:
        FARs = np.array(FARs)
        num_false_alarms = np.round(num_neg * FARs).astype(np.int32)

        thresholds = []
        for num_false_alarm in num_false_alarms:
            if num_false_alarm==0:
                threshold = score_neg[0] + epsilon
            else:
                threshold = score_neg[num_false_alarm-1]
            thresholds.append(threshold)
        thresholds = np.array(thresholds)

    return thresholds


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def l2_norm(input, axis=1):
    """l2 normalize
    """
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output, norm


def fuse_features_with_norm(stacked_embeddings, stacked_norms, fusion_method='norm_weighted_avg'):

    assert stacked_embeddings.ndim == 3 # (n_features_to_fuse, batch_size, channel)
    if stacked_norms is not None:
        assert stacked_norms.ndim == 3 # (n_features_to_fuse, batch_size, 1)
    else:
        assert fusion_method not in ['norm_weighted_avg', 'pre_norm_vector_add']

    if fusion_method == 'norm_weighted_avg':
        weights = stacked_norms / stacked_norms.sum(dim=0, keepdim=True)
        fused = (stacked_embeddings * weights).sum(dim=0)
        fused, _ = l2_norm(fused, axis=1)
        fused_norm = stacked_norms.mean(dim=0)
    elif fusion_method == 'pre_norm_vector_add':
        pre_norm_embeddings = stacked_embeddings * stacked_norms
        fused = pre_norm_embeddings.sum(dim=0)
        fused, fused_norm = l2_norm(fused, axis=1)
    elif fusion_method == 'average':
        fused = stacked_embeddings.sum(dim=0)
        fused, _ = l2_norm(fused, axis=1)
        if stacked_norms is None:
            fused_norm = torch.ones((len(fused), 1))
        else:
            fused_norm = stacked_norms.mean(dim=0)
    elif fusion_method == 'concat':
        fused = torch.cat([stacked_embeddings[0], stacked_embeddings[1]], dim=-1)
        if stacked_norms is None:
            fused_norm = torch.ones((len(fused), 1))
        else:
            fused_norm = stacked_norms.mean(dim=0)
    elif fusion_method == 'faceness_score':
        raise ValueError('not implemented yet. please refer to https://github.com/deepinsight/insightface/blob/5d3be6da49275602101ad122601b761e36a66a01/recognition/_evaluation_/ijb/ijb_11.py#L296')
        # note that they do not use normalization afterward.
    else:
        raise ValueError('not a correct fusion method', fusion_method)

    return fused, fused_norm


def infer(model, dataloader, use_flip_test, fusion_method):
    model.eval()
    features = []
    norms = []
    with torch.no_grad():
        for images, idx in tqdm(dataloader):

            feature = model(images.to("cuda:0"))
            if isinstance(feature, tuple):
                feature, norm = feature
            else:
                norm = None

            if use_flip_test:
                fliped_images = torch.flip(images, dims=[3])
                flipped_feature = model(fliped_images.to("cuda:0"))
                if isinstance(flipped_feature, tuple):
                    flipped_feature, flipped_norm = flipped_feature
                else:
                    flipped_norm = None

                stacked_embeddings = torch.stack([feature, flipped_feature], dim=0)
                if norm is not None:
                    stacked_norms = torch.stack([norm, flipped_norm], dim=0)
                else:
                    stacked_norms = None

                fused_feature, fused_norm = fuse_features_with_norm(stacked_embeddings, stacked_norms, fusion_method=fusion_method)
                features.append(fused_feature.cpu().numpy())
                norms.append(fused_norm.cpu().numpy())
            else:
                features.append(feature.cpu().numpy())
                norms.append(norm.cpu().numpy())

    features = np.concatenate(features, axis=0)
    norms = np.concatenate(norms, axis=0)
    return features, norms

def load_pretrained_model(model_name='ir50'):
    # load model and pretrained statedict
    ckpt_path = adaface_models[model_name][0]
    arch = adaface_models[model_name][1]

    model = net.build_model(arch)
    statedict = torch.load(ckpt_path)['state_dict']
    model_statedict = {key[6:]:val for key, val in statedict.items() if key.startswith('model.')}
    model.load_state_dict(model_statedict)
    model.eval()
    return model

#def load_pretrained_model(model_name='ir50'):
#     # load model and pretrained statedict
#     model = resnet18()
#     checkpoint = torch.load('/workspace/codes/resnet18-5c106cde.pth')
#     mystatedict = {}
#     for key in checkpoint.keys():
#         m = key.replace("module.","")
#         mystatedict[m] = checkpoint[key]
#     model.load_state_dict(mystatedict, strict=False)

#     model = torch.nn.DataParallel(model)
#     model.eval()
#     return model

    


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='tinyface')

    parser.add_argument('--data_root', default='./val_data') 
    parser.add_argument('--gpu', default=0, type=int, help='gpu id')
    parser.add_argument('--batch_size', default=32, type=int, help='')
    parser.add_argument('--model_name', type=str, default='ir18_casia')
    parser.add_argument('--use_flip_test', type=str2bool, default='True')
    parser.add_argument('--fusion_method', type=str, default='average', choices=('average', 'norm_weighted_avg', 'pre_norm_vector_add', 'concat', 'faceness_score'))
    parser.add_argument('--arch', default='iresnet100', type=str, help='backbone architechture')
    parser.add_argument('--embedding_size', default=512, type=int, help='The embedding feature size')
    parser.add_argument('--resume', default=None, type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
    parser.add_argument('--cpu-mode', action='store_true', help='Use the CPU.')
    args = parser.parse_args()

    # load model
    adaface_models = {
        'ir50': ["../pretrained/adaface_ir50_ms1mv2.ckpt", 'ir_50'],
        'ir101_ms1mv2': ["../pretrained/adaface_ir101_ms1mv2.ckpt", 'ir_101'],
        'ir101_ms1mv3': ["../pretrained/adaface_ir101_ms1mv3.ckpt", 'ir_101'],
        'ir101_webface4m': ["/workspace/codes/CurricularFace/adaface_ir101_webface4m.ckpt", 'ir_101'],
        'ir101_webface12m': ["../pretrained/adaface_ir101_webface12m.ckpt", 'ir_101'],
        'ir18_casia': ["/workspace/sadiq/MagFace/magface_iresnet100_quality.pth", 'iresnet100']
    }
    
    assert args.model_name in adaface_models
    checkpoint = torch.load('/workspace/codes/quickmatch/backbone.pth')
    # load model
    model = builder_inf(args)
    model = torch.nn.DataParallel(model)
    # model = get_model("r100", fp16=False)
    # model.load_state_dict(checkpoint)

    # model = load_pretrained_model(args.model_name)
    model.to('cuda:{}'.format(args.gpu))

    tinyface_test = TinyFaceTest(tinyface_root=args.data_root,
                                                 alignment_dir_name='aligned_pad_0.1_pad_high')

    # set save root
    gpu_id = args.gpu
    save_path = os.path.join('./tinyface_result', args.model_name, "fusion_{}".format(args.fusion_method))

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    print('save_path: {}'.format(save_path))

    img_paths = tinyface_test.image_paths
    print('total images : {}'.format(len(img_paths)))
    dataloader = data_utils.prepare_dataloader(img_paths,  args.batch_size, num_workers=0)
    features, norms = infer(model, dataloader, use_flip_test=args.use_flip_test, fusion_method=args.fusion_method)
    results = tinyface_test.test_identification(features, ranks=[1,5,20])
    print(results)
    pd.DataFrame({'rank':[1,5,20], 'values':results}).to_csv(os.path.join(save_path, 'result.csv'))
