from cross_validation import *
from prepare_data_DEAP import *
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ######## Data ########
    parser.add_argument('--dataset', type=str, default='DEAP')
    parser.add_argument('--data-path', type=str,
                        default=r'.\data_result\with_base_0.5')
    parser.add_argument('--subjects', type=int, default=32)
    parser.add_argument('--num_class', type=int, default=2, choices=[2, 3, 4])
    parser.add_argument('--label-type', type=str, default='V', choices=['A', 'V'])
    parser.add_argument('--segment', type=int, default=0.5, help='segment length in seconds')
    parser.add_argument('--trial-duration', type=int, default=0.5, help='trial duration in seconds')
    parser.add_argument('--overlap', type=float, default=0)
    parser.add_argument('--sampling-rate', type=int, default=128)
    parser.add_argument('--input-shape', type=tuple, default=(1, 28, 512))
    parser.add_argument('--data-format', type=str, default='raw')
    parser.add_argument('--e_layers', type=int, default=1, help='num of encoder layers')
    parser.add_argument('--top_k', type=int, default=2, help='for TimesBlock')
    ######## Training Process ########
    parser.add_argument('--random-seed', type=int, default=1234)
    parser.add_argument('--max-epoch', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0.2)

    parser.add_argument('--save-path', default='./save/')
    parser.add_argument('--load-path', default='./save/max-acc.pth')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--save-model', type=bool, default=True)
    ######## Model Parameters ########
    parser.add_argument('--model', type=str, default='MT_timenets',
                        help='TSception or MT_timenets or MLP or LogisticRegression or lstm or eegnet')


    parser.add_argument('--T', type=int, default=15)
    parser.add_argument('--graph-type', type=str, default='TS', choices=['TS', 'O'], 
                        help='TS for the channel order of TSception, O for the original channel order')
    parser.add_argument('--hidden', type=int, default=32)

    ######## Reproduce the result using the saved model ######
    parser.add_argument('--reproduce', type=bool, default=False)

    parser.add_argument('--experiment', type=str, choices=['Idex', 'Dedx'], default='Dedx',
                        help='independent dependence experiment')
    args = parser.parse_args()

    sub_to_run = np.arange(args.subjects)

    pd = PrepareData(args)
    pd.run(sub_to_run, split=True, feature=False, expand=False)

    if args.experiment == 'Dedx':
        cv = CrossValidation(args)
        seed_all(args.random_seed)
        cv.dependent(subject=sub_to_run, fold=5, reproduce=args.reproduce)  # To do leave one trial out please set fold=40
    elif args.experiment == 'Idex':
        cv = CrossValidation(args)
        seed_all(args.random_seed)
        cv.independence(subject=sub_to_run, fold=10, reproduce=args.reproduce)
