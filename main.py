import csv
import math
import time
import torch
import parser
import argparse

import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as Variable
from tensorboardX import SummaryWriter

from data import *
from model import *
from util import *
from train_han import *
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score


def build_model(args, embeddings=None):
    mil = HAN(args)
    if args.emb:
        mil.set_embedding(embeddings) #TODO:
    return mil

def main(args):

    if args.cuda:
        torch.cuda.set_device(args.gpu_id)
    log_level = logging.INFO
    # formatter_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    if not args.logfile:
        logging.basicConfig(stream=sys.stdout, format=formatter_str, level=log_level)
    else:
        logging.basicConfig(filename=args.logfile, level=log_level)

    logger = logging.getLogger("MIL")

    writer = SummaryWriter()

    logger.info('Building corpus....')
    corpus = Corpus(args)
    tr_batcher = Batchify(corpus.train, corpus.train_labels, corpus.train_lens, bsz=args.batch_size,  cuda=args.cuda)
    val_batcher = Batchify(corpus.val, corpus.val_labels, corpus.val_lens,  bsz=args.eval_size, cuda=args.cuda)
    # test_batcher = Batchify(corpus.test, corpus.test_labels, corpus.test_lens, cuda=args.cuda)

    # Define Modelt
    ntokens = len( corpus.dictionary )
    nclass = args.nclass

    embeddings=None
    if args.emb:
        logger.info('Loading word embeddings, this could take a while...')
        embeddings = get_embeddings(args.emb_path, corpus.dictionary, args.emb_dim)

    if not args.pretrained:
        model = build_model(args, embeddings)
    else:
        model = torch.load(args.pretrained)
        print('Pretrained model loaded.')

    if args.nclass==1: #binary classification
        criterion = nn.BCELoss()
    else:
        criterion = nn.CrossEntropyLoss()

    if args.cuda:
        model.cuda()
        criterion.cuda()

    # Training Process
    logger.info('Start training...')
    lr = float(args.lr)
    best_val_loss = None
    train_losses = []
    print('# of Epochs:', args.epochs)

    # main training loop
    for epoch in range( 1, args.epochs + 1 ):
        epoch_start_time = time.time()
        train_losses.append( train( model, criterion, tr_batcher, epoch, writer, logger, args ) )
        val_loss, pre, rec, f1, acc, targets, preds = evaluate( model, criterion, val_batcher, epoch, writer, logger, args )
        # Save the best model and Anneal the learning rate.
        if not best_val_loss or val_loss < best_val_loss:  # save best model accross folds
            best_val_loss = val_loss
            patience_cnt = 0
            with open( args.save, 'wb+' ) as f:
                torch.save( model , f )
        else:
            patience_cnt += 1
            args.lr /= 2.0
            if patience_cnt == args.patience:  #stop training if val loss hasn't increased for 5 epochs
                print('Early stopping after {} epochs.'.format(epoch))
                break;

    report  = classification_report(targets, preds, output_dict=True)
    if not os.path.exists(os.path.join(args.stats_dir, args.experiment+'/EVAL')):
        os.makedirs(os.path.join(args.stats_dir,  args.experiment+'/EVAL'))
    experiment_path = os.path.join(args.stats_dir, args.experiment+'/EVAL')
    with open(os.path.join(experiment_path, 'eval.json'), 'w+') as f:
        json.dump(report, f)

    writer.export_scalars_to_json("./all_scalars.json")
    writer.close()

    test_batcher = Batchify(corpus.test, corpus.test_labels, corpus.test_lens, bsz=args.batch_size, cuda=args.cuda)
    test_loss, pre, rec, f1, acc, targets, preds = evaluate( model, criterion, test_batcher, epoch, writer, logger, args, mode='TEST')
    report  = classification_report(targets, preds, output_dict=True)
    report['accuracy'] = accuracy_score(targets, preds)
    if not os.path.exists(os.path.join(args.stats_dir, args.experiment+'/TEST')):
        os.makedirs(os.path.join(args.stats_dir,  args.experiment+'/TEST'))
    experiment_path = os.path.join(args.stats_dir, args.experiment+'/TEST')
    with open(os.path.join(experiment_path, 'test.json'), 'w+') as f:
        json.dump(report, f)

    print( '=' * 80 )
    print( '| End of training | test loss {} |'.format( test_loss ) )
    print( '=' * 80 )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    # train
    parser.add_argument('--epochs', type=int, default=50, help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--eval_size', type=int, default=16, help='Evaluation batch size')
    parser.add_argument('--seed', type=int, default=3242, help='random seed')
    parser.add_argument('--pretrained', type=str, default='', help='whether start from pretrained model')
    parser.add_argument('--cuda', default=True, action='store_true', help='use CUDA')
    parser.add_argument('--log_interval', type=int, default=10, help='report interval')
    parser.add_argument('--logfile', type=str, default='./.log_analysis', help='log to file')
    parser.add_argument('--save', type=str, default='/home/sudo777/mercury/output/analyses/hansa_35k_new.pt', help='path to save the final model')
    parser.add_argument('--lr', type=float, default=0.05, help='Evaluation batch size')
    parser.add_argument('--momentum', type=float, default=0.9, help='sgd with momentum')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight_decay')
    parser.add_argument('--folds', type=int, default=5, help='number of cross validation folds')
    parser.add_argument('--gpu_id', type=str, default=2, help='')
    parser.add_argument('--patience', type=int, default=7, help='patience for early stopping')
    parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping')


    #data
    parser.add_argument('--data_dir', type=str, default='/home/sudo777/mercury/data_autogsr/english_15/', help='Location of the data corpus')
    parser.add_argument('--emb_path', type=str, default='/home/sudo777/mercury/data_autogsr/english_15/embeddings/word2vec.100d.35k.w2v', help='path to the embedding file')
    parser.add_argument('--from_emb', default=True, action='store_true', help='Whether build vocab from embedding file')
    parser.add_argument('--vocab_size', type=int, default=50000, help='vocab size')
    parser.add_argument('--stop_words', type=str, default='/home/sudo777/mercury/data_autogsr/en_stopwords.txt', help='Location of the data corpus')
    parser.add_argument('--emb', default=True, action='store_true', help='Whether to load pretrained word embddings')

    #stats
    parser.add_argument('--experiment', type=str, default='CUEn/analyses/HANSA_35k_new', help='Location of the data corpus')
    parser.add_argument('--stats_dir', type=str, default='/home/sudo777/mercury/stats', help='location to save classification reports')
    #model
    parser.add_argument('--emb_dim', type=int, default=100, help='Location of the data corpus')
    parser.add_argument('--max_sent', type=int, default=15, help='Max number of sents in a doc')
    parser.add_argument('--max_word', type=int, default=70, help='Max number of words in a sent')
    parser.add_argument('--num_kernels', type=int, default=3, help='Evaluation batch size')
    parser.add_argument('--num_filters', type=int, default=100, help='random seed')
    parser.add_argument('--dropout', type=float, default=0.4, help='whether start from pretrained model')
    parser.add_argument('--nclass', type=int, default=2, help='use CUDA')
    parser.add_argument('--mlp_nhid', type=int, default=512, help='Number of hidden units in the sentence classifier') # prev 128
    parser.add_argument('--lstm_layer', type=int, default=1, help='Evaluation batch size')
    parser.add_argument('--lstm_hidden', type=int, default=50, help='Evaluation batch size')
    parser.add_argument('--n_layers', type=int, default=1, help='Number of GRU stacking layers')
    parser.add_argument('--bidirectional', default=True, action='store_true', help='Evaluation batch size')
    parser.add_argument('--alpha', type=float,  default=0.5, help='constant controlling instance ratio' )
    parser.add_argument('--beta', type=float,  default=0.5, help='constant controlling instance ratio' )
    parser.add_argument('--m0', type=float,  default=0.5, help='margin for instance loss' )
    parser.add_argument('--p0', type=float,  default=0.6, help='instance loss constant' )
    parser.add_argument('--K', type=int,  default=3, help='Number of sentences to select' )
    parser.add_argument('--da', type=int, default=350, help='Hidden dimension for self attention')
    parser.add_argument('--aspects', type=int, default=5, help='Number of aspects')


    args = parser.parse_args()
    main(args)
