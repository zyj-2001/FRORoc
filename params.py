import argparse

def ParseArgs():
	parser = argparse.ArgumentParser(description='Model Params')
	parser.add_argument('--lr', default=8e-4, type=float, help='learning rate')
	parser.add_argument('--eta_min', default=1e-6, type=float, help='minimum learning rate for cosine scheduler')
	parser.add_argument('--batch', default=1024, type=int, help='batch size')
	parser.add_argument('--tstBat', default=256, type=int, help='number of users in a testing batch')
	parser.add_argument('--reg', default=1e-5, type=float, help='weight decay regularizer')
	parser.add_argument('--epoch', default=50, type=int, help='number of epochs')
	parser.add_argument('--latdim', default=512, type=int, help='embedding size')
	parser.add_argument('--gnn_layer', default=1, type=int, help='number of gnn layers')
	parser.add_argument('--topk', default=20, type=int, help='K of top K')
	parser.add_argument('--data', default='baby', choices=('baby', 'sports', 'tiktok'), help='name of dataset')
	parser.add_argument('--ssl_reg', default=1e-2, type=float, help='weight for contrative learning')
	parser.add_argument('--temp', default=0.5, type=float, help='temperature in contrastive learning')
	parser.add_argument('--tstEpoch', default=1, type=int, help='number of epoch to test while training')
	parser.add_argument('--gpu', default='0', type=str, help='indicates which gpu to use')
	parser.add_argument("--seed", type=int, default=42, help="random seed")

	parser.add_argument('--keepRate', default=0.5, type=float, help='ratio of edges to keep')
	
	parser.add_argument('--dims', type=str, default='[1000]')
	parser.add_argument('--d_emb_size', type=int, default=10)
	parser.add_argument('--norm', action='store_true')
	parser.add_argument('--steps', type=int, default=10, help='ODE integration steps for flow matching inference')
	parser.add_argument('--noise_scale', type=float, default=0.1, help='std of initial noise for flow matching')
	parser.add_argument('--scheduler', type=str, default='cosine',
		help='fallback scheduler for all modalities: cosine | linear | vp | linear_vp | poly2 | poly3')
	parser.add_argument('--scheduler_image', type=str, default=None,
		help='flow path scheduler for image (default: same as --scheduler)')
	parser.add_argument('--scheduler_text',  type=str, default=None,
		help='flow path scheduler for text  (default: same as --scheduler)')
	parser.add_argument('--scheduler_audio', type=str, default=None,
		help='flow path scheduler for audio (default: same as --scheduler)')
	parser.add_argument('--ode_method', type=str, default='midpoint',
		help='ODE integration method: euler | midpoint (2nd-order) | rk4 (4th-order)')

	parser.add_argument('--rebuild_k', type=int, default=1)
	parser.add_argument('--e_loss', type=float, default=0.1)
	parser.add_argument('--ris_lambda', type=float, default=0.5)
	parser.add_argument('--ris_adj_lambda', type=float, default=0.2)
	parser.add_argument('--trans', type=int, default=0, help='0: R*R, 1: Linear, 2: allrecipes')
	parser.add_argument('--cl_method', type=int, default=0, help='0:m vs m ; 1:m vs main')

	parser.add_argument('--clear_rank', type=int, default=0,
		help='CLEAR null-space rank r (0 = disabled); removes top-r shared SVD directions between image & text')
	parser.add_argument('--clear_lam', type=float, default=1.0,
		help='CLEAR projection strength λ (1.0 = full removal)')
	parser.add_argument('--clear_text', action='store_true', default=False,
		help='also project text to its null-space (default: text-first, keep text full)')

	return parser.parse_args()
args = ParseArgs()
