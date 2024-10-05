from argparse import ArgumentParser
import torch

HELP = "the early stop threshold for assignment update stage"


def config_parser() -> ArgumentParser:
    """
    Generate a generic parser
    """
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=2024, help='random seed')
    parser.add_argument('--bpr_batch', type=int, default= 32768,
                        help="the batch size for bpr loss training procedure")#16384 #32768 #65536 #131072 #8192
    parser.add_argument('--latent_dim', type=int, default=128,
                        help="the dimension size of each embedding vector")
    parser.add_argument('--l2_penalty_factor', type=float, default=5,
                        help="the penalty factor for L2 regularization. Actual penalty weight = factor * 1e-4")
    parser.add_argument('--num_layers', type=int, default=4,
                        help="the layer num of lightGCN")
    parser.add_argument('--lr', type=float, default=0.003,
                        help="the learning rate")
    parser.add_argument("--optimizer_weight_decay", type=float, default=0.,
                        help="The weight decay (l2 regularization) on Optimizer.")
    parser.add_argument('--decay', type=float, default=1e-4,
                        help="the weight decay for l2 normalization")
    parser.add_argument('--test_batch', type=int, default=20,
                        help="the batch size of users for testing")
    parser.add_argument("--neg_size", type=int, default=5,
                        help="The number of negative items sampled for each (user, pos item) pair.") #5
    parser.add_argument('--dataset_name', type=str, default='yelp2020',
                        choices=["gowalla", "yelp2020"],
                        help="available datasets: [gowalla, yelp2020]")
    parser.add_argument("--res_prepath", default="./logs/",
                        help='The prepended path for result saving.')
    parser.add_argument("--device_id", default=0 if torch.cuda.is_available() else "cpu",
                        help="The CUDA device to be used")
    parser.add_argument('--epochs', type=int, default=900, help='Max training epochs')
    parser.add_argument("--early_stop_fixed_assignment_stage", type=int, default=5,
                        help="the early stop threshold for fixed assignment training stage")
    parser.add_argument("--early_stop_assignment_update_stage", type=int, default=5,
                        help=("%s" % HELP))
    parser.add_argument("--ks", nargs="+", help="The range of k value for evaluation",
                        default=[5, 10, 20, 50], type=int)
    parser.add_argument("--additional_alias", type=str, default=None,
                        help="any additional text to be appended to alias")
    parser.add_argument("--max_grad_value", type=float, default=3,
                        help="The max grad value possible. Clip all grads > 3 or < 3 to avoid gradient explosion.")
    parser.add_argument("--num_clusters", type=int, default=300,
                        help="Number of clusters used in the system, if pretrain is not executed.")
    parser.add_argument("--num_composition_centroid", type=int, default=2,
                        help="The number of predicted top-k centroid embeddings used to generate an entity's"
                             "embedding.")
    parser.add_argument("--use_metis_init", type=lambda x: x.lower() == "metis", default=True,
                        help="whether use random initialization or METIS for assignment matrix")
    parser.add_argument("--assignment_update_frequency", type=str, default="every-epoch",
                        help='the frequency of assignment update. Possible options: only-once, every-epoch, every-k-epochs')

    ####################################################################
    parser.add_argument("--sign_ft", type=int, default=1,
                        help="sign 0 for finetune stage. 1 for finetune stage")
    parser.add_argument("--init_path_ft", default="./logs/amazon-book_latent_dim_128_num_cluster_300_seed_2020_num_composition_embs_2_lr_3e-03_optimizer_weight_decay_0.0_l2_penalty_5",
                        help='The path of load the model from coarse stage for finetune stage.')
    parser.add_argument("--threshold_ft", type=float, default=1,
                        help="Soft Thresholding threshold")
    parser.add_argument("--gate", type=float, default=0.2,
                        help="Weight factor for coarse-to-fine transition")
    parser.add_argument("--top_k_ft", type=int, default=4,
                        help="attention score map top_k_ft for finetune stage.")
    parser.add_argument("--n_components_ft", type=int, default=80,
                        help="n_components for SparsePCA")
    parser.add_argument("--num_clusters_ft", type=int, default=100,
                        help="Number of clusters for finetune stage used in the system, if pretrain is not executed.")
    return parser
