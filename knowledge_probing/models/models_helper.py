from transformers import BertConfig, AutoTokenizer, BertModel, BertForMaskedLM
from knowledge_probing.models.lightning.decoder import Decoder
from knowledge_probing.models.lightning.hugging_decoder import HuggingDecoder
from knowledge_probing.file_utils import find_checkpoint_in_dir


def get_model(args):
    # Get config for Decoder
    config = BertConfig.from_pretrained(args.bert_model_type)
    config.output_hidden_states = True

    # Load Bert as BertModel which is plain and has no head on top
    if args.use_model_from_dir:
        bert = BertModel.from_pretrained(args.model_dir, config=config)
    else:
        bert = BertModel.from_pretrained(args.bert_model_type, config=config)

    # Make sure the bert model is not trained
    bert.eval()
    bert.requires_grad = False
    for param in bert.parameters():
        param.requires_grad = False
    bert.to(args.device)

    # Get the right decoder
    if args.decoder_type == "Decoder":
        decoder = Decoder(hparams=args, bert=bert, config=config)
    else:
        if saved_model_has_mlm_head(args.model_dir):
            print('Using models own mlm head')
            mlm_head = (BertForMaskedLM.from_pretrained(
                args.model_dir, config=config)).cls
            decoder = HuggingDecoder(
                hparams=args, bert=bert, config=config, decoder=mlm_head)
        else:
            # Initialize with standard pre-trained mlm head
            decoder = HuggingDecoder(hparams=args, bert=bert, config=config)

    return decoder


def saved_model_has_mlm_head(path):
    if path is not None:
        config = BertConfig.from_pretrained(path)

        if config.architectures[0] == 'BertForMaskedLM':
            return True

    return False


def load_best_model_checkpoint(decoder, args):
    checkpoint_file = find_checkpoint_in_dir(args.decoder_save_dir)

    print('Loading best checkpoint: {}'.format(checkpoint_file))

    if args.decoder_type == "Decoder":
        best_model = Decoder.load_from_checkpoint(
            checkpoint_file, hparams=args, bert=decoder.bert, config=decoder.config)
    else:
        best_model = HuggingDecoder.load_from_checkpoint(
            checkpoint_file, hparams=args, bert=decoder.bert, config=decoder.config)

    return best_model
