import argparse
import sys
import torch as th
from copy import copy


from utils.configuration import Configuration
from model.scripts import training, evaluation
from data.datasets.moving_mnist.dataset import MovingMNISTDataset
from data.datasets.video.dataset import VideoDataset, MultipleVideosDataset
from data.datasets.CATER.dataset import CaterDataset, CaterLatentDataset
from data.datasets.CLEVRER.dataset import ClevrerDataset

CFG_PATH = "cfg.json"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-cfg", default=CFG_PATH)
    parser.add_argument("-num-gpus", default=1, type=int)
    parser.add_argument("-n", default=-1, type=int)
    parser.add_argument("-load", default="", type=str)
    parser.add_argument("-dataset-file", default="", type=str)
    parser.add_argument("-device", default=0, type=int)
    parser.add_argument("-testset", action="store_true")
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("-train", action="store_true")
    mode_group.add_argument("-eval", action="store_true")
    mode_group.add_argument("-save", action="store_true")
    mode_group.add_argument("-export", action="store_true")
    parser.add_argument("-objects", action="store_true")
    parser.add_argument("-nice", action="store_true")
    parser.add_argument("-individual", action="store_true")

    args = parser.parse_args(sys.argv[1:])

    if not args.objects and not args.nice and not args.individual:
        args.objects = True

    cfg = Configuration(args.cfg)

    if args.device >= 0:
        cfg.device = args.device
        cfg.model_path = f"{cfg.model_path}.device{cfg.device}"

    if args.n >= 0:
        cfg.device = args.device
        cfg.model_path = f"{cfg.model_path}.run{args.n}"

    num_gpus = th.cuda.device_count()
    
    if cfg.device >= num_gpus:
        cfg.device = num_gpus - 1

    if args.num_gpus > 0:
        num_gpus = args.num_gpus

    print(f'Using {num_gpus} GPU{"s" if num_gpus > 1 else ""}')
    print(f'{"Training" if args.train else "Evaluating"} model {cfg.model_path}')

    trainset = None
    valset   = None
    testset  = None
    if cfg.datatype == "moving_mnist":
        trainset = MovingMNISTDataset(2, 'train', 64, 64, cfg.sequence_len)
        valset   = MovingMNISTDataset(2, 'train', 64, 64, cfg.sequence_len)
        testset  = MovingMNISTDataset(2, 'test', 64, 64, cfg.sequence_len)

    if cfg.datatype == "video" or cfg.datatype == "multiple-videos":
        if args.run_testset:
            cfg.sequence_len = 3

        if args.save_patch:
            cfg.sequence_len = 0

        if args.save:
            cfg.sequence_len = 1
            cfg.model.batch_size = 1

        if args.save_patch or args.eval_patch:
            cfg.model.latent_size[0] = cfg.model.patch_grid_size[0] * 2
            cfg.model.latent_size[1] = cfg.model.patch_grid_size[1] * 2

    if cfg.datatype == "video":
        trainset = None if args.save and args.testset else VideoDataset("./", cfg.dataset, "train", (cfg.model.latent_size[1] * 2**(cfg.model.level*2), cfg.model.latent_size[0] * 2**(cfg.model.level*2)), cfg.sequence_len + 1)
        valset   = None if args.save else VideoDataset("./", cfg.dataset, "test",  (cfg.model.latent_size[1] * 2**(cfg.model.level*2), cfg.model.latent_size[0] * 2**(cfg.model.level*2)), cfg.sequence_len + 1)
        testset  = VideoDataset("./", cfg.dataset, "test", (cfg.model.latent_size[1] * 2**(cfg.model.level*2), cfg.model.latent_size[0] * 2**(cfg.model.level*2)), cfg.sequence_len + 1)
        cfg.sequence_len += 1

    if cfg.datatype == "multiple-videos":
        trainset = None if args.save and args.testset else MultipleVideosDataset("./", cfg.dataset, "train", (cfg.model.latent_size[1] * 2**(cfg.model.level*2), cfg.model.latent_size[0] * 2**(cfg.model.level*2)), cfg.sequence_len + 1)
        valset   = None if args.save else MultipleVideosDataset("./", cfg.dataset, "test",  (cfg.model.latent_size[1] * 2**(cfg.model.level*2), cfg.model.latent_size[0] * 2**(cfg.model.level*2)), cfg.sequence_len + 1)
        testset  = MultipleVideosDataset("./", cfg.dataset, "test", (cfg.model.latent_size[1] * 2**(cfg.model.level*2), cfg.model.latent_size[0] * 2**(cfg.model.level*2)), cfg.sequence_len + 1)
        cfg.sequence_len += 1

    if cfg.datatype == "clevrer":
        trainset = None if args.save and args.testset else ClevrerDataset("./", cfg.dataset, "train", (cfg.model.latent_size[1] * 2**(cfg.model.level*2), cfg.model.latent_size[0] * 2**(cfg.model.level*2)))
        valset   = None if args.save else ClevrerDataset("./", cfg.dataset, "val",   (cfg.model.latent_size[1] * 2**(cfg.model.level*2), cfg.model.latent_size[0] * 2**(cfg.model.level*2)))
        testset  = ClevrerDataset("./", cfg.dataset, "test",  (cfg.model.latent_size[1] * 2**(cfg.model.level*2), cfg.model.latent_size[0] * 2**(cfg.model.level*2)))

        valset.train  = False
        testset.train = False
        
        cfg.sequence_len += 1

    if cfg.datatype == "cater":
        trainset = None if args.save and args.testset else CaterDataset("./", cfg.dataset, "train", (cfg.model.latent_size[1] * 2**(cfg.model.level*2), cfg.model.latent_size[0] * 2**(cfg.model.level*2)))
        valset   = None if args.save else CaterDataset("./", cfg.dataset, "val",   (cfg.model.latent_size[1] * 2**(cfg.model.level*2), cfg.model.latent_size[0] * 2**(cfg.model.level*2)))
        testset  = CaterDataset("./", cfg.dataset, "test",  (cfg.model.latent_size[1] * 2**(cfg.model.level*2), cfg.model.latent_size[0] * 2**(cfg.model.level*2)))

        
        cfg.sequence_len += 1

    if cfg.datatype == "latent-cater":
        if args.dataset_file != "":
            cfg.dataset = args.dataset_file
        trainset = CaterLatentDataset("./", cfg.dataset, "train")
        valset   = CaterLatentDataset("./", cfg.dataset, "val")
        testset  = CaterLatentDataset("./", cfg.dataset, "test")

    if cfg.datatype == "latent-cater":
        if cfg.latent_type == "snitch_tracker":
            training.train_latent_tracker(cfg, trainset, valset, testset, args.load)
        elif cfg.latent_type == "object_behavior":
            training.train_latent_action_classifier(cfg, trainset, testset, args.load)
    elif args.train:
        training.run(cfg, num_gpus, trainset, valset, testset, args.load, (cfg.model.level*2))
    elif args.eval:
        evaluation.evaluate(cfg, num_gpus, testset if args.testset else valset, args.load, (cfg.model.level*2))
    elif args.save:
        evaluation.save(cfg, testset if args.testset else trainset, args.load, (cfg.model.level*2), cfg.model.input_size, args.objects, args.nice, args.individual)
    elif args.export:
        evaluation.export_dataset(cfg, trainset, testset, args.load, f"{args.load}.latent-states")
