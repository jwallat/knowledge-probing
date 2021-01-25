import os
import torch
import streamlit as st
from knowledge_probing.models.lightning.hugging_decoder import HuggingDecoder
from transformers import AutoTokenizer, BertConfig, BertModel
from knowledge_probing.datasets.cloze_data_utils import topk
from dotmap import DotMap


def main():
    st.title('Prediction Explorer')
    # MODEL
    st.subheader('Model Selection')

    bert_model_type = st.selectbox(
        'Select model type', ['bert-base-uncased', 'bert-base-cased'], 0)

    st.write('When querying a model, you have the option to either use a standard BERT model or select your own model with a trained masked language modeling head')

    use_own_model = st.checkbox('Use own model or MLM head', False)
    # use_default = st.checkbox('Use default model', not use_own_model)

    if use_own_model:
        st.subheader('Own Model Selection')
        st.write('When using own models there are multiple options. If you want to use the standard pre-trained BERT model with a different MLM head, just select the checkbox and supply the path to your MLM head. ')
        use_own_bert = st.checkbox('Use own BERT', False)
        if use_own_bert:
            bert_model_path = st.text_input('Fine-tuned model path')
        else:
            bert_model_path = None

        use_own_decoder = st.checkbox('Use trained MLM head', False)
        if use_own_decoder:
            trained_decoder_path = st.text_input(
                'Trained decoder path', '/home/jonas/git/knowledge-probing/data/outputs/_uncased_trained-True_Huggingface_pretrained_decoder_21_7_2020__17-10/decoder/epoch=1.ckpt')
        else:
            trained_decoder_path = None

        decoder = get_model(
            bert_model_type, decoder_path=trained_decoder_path, model_dir=bert_model_path)

    else:
        decoder = get_model(bert_model_type)

    st.write('--> Selected model: ')
    if not use_own_model:
        st.write('Standard pre-trained BERT ' + bert_model_type)
    else:
        if use_own_bert and use_own_decoder:
            st.write('Own BERT and MLM head as supplied')
        elif use_own_bert and not use_own_decoder:
            st.write('Own BERT and pre-trained MLM head')
        elif not use_own_bert and use_own_decoder:
            st.write('standard pre-trained BERT and own MLM head')
        else:
            st.write('Standard pre-trained BERT ' + bert_model_type)
    # INPUT
    st.subheader('Input')
    masked_sentence = st.text_input(
        "Masked Sentence", 'The capital of France is [MASK].')
    assert '[MASK]' in masked_sentence

    num_predictions = st.number_input('Number predictions to show', 10)

    probing_layer = st.slider('Probing layer', 1, 12, 12)

    # OUTPUT
    st.subheader('Outputs')
    # Show predictions
    predictions, _ = get_predictions(
        masked_sentence, decoder, layer=probing_layer)
    # st.write('Predicted sentence:')
    # st.write(predicted_sentence)
    st.write('Predicted tokens for [MASK]:')
    st.write(predictions[:num_predictions])

    answer_token = st.text_input(
        "Find token in predictions:", 'london')

    try:
        # print(answer_token)
        # print(len(predictions))
        # st.write(predictions[:5])
        # first = predictions[:5]
        # print(first)
        # print(first.index(answer_token))
        rank = predictions.index(str(answer_token))
        st.write('--> Ranked at position: ' + str(rank))

    except Exception as e:
        print(str(e))
        st.write('Could not find the token in the predictions')


@st.cache
def get_predictions(masked_sentence, decoder, layer):
    tokenizer: AutoTokenizer = decoder.tokenizer

    input_ids = tokenizer.encode_plus(
        masked_sentence, add_special_tokens=True, return_tensors='pt')['input_ids']
    print(input_ids)
    input_list = input_ids.numpy().tolist()
    print(input_list)

    mask_index = input_ids.numpy().tolist()[0].index(tokenizer.mask_token_id)

    outputs = decoder(input_ids, masked_lm_labels=input_ids, layer=layer)

    prediction_scores = outputs[1]

    # st.write('Predicted sentence: {}'.format(
    #     predicted_sentence(prediction_scores, tokenizer)))
    sentence = decode_predictions(prediction_scores, tokenizer)
    # st.write('Predicted sentence (decoded): {}'.format(sentence))

    topk_tokens = topk(prediction_scores,
                       mask_index, k=len(tokenizer), tokenizer=tokenizer)

    return topk_tokens, sentence


def predicted_sentence(scores, tokenizer):
    predicted_ids = torch.reshape(torch.argmax(scores, dim=2), (-1,))
    predicted_tokens = tokenizer.convert_ids_to_tokens(predicted_ids)

    return predicted_tokens


def decode_predictions(scores, tokenizer):
    predicted_ids = torch.reshape(torch.argmax(scores, dim=2), (-1,))
    predicted_tokens = tokenizer.decode(predicted_ids)

    return predicted_tokens


@st.cache
def get_model(bert_model_type, decoder_path=None, model_dir=None):
    # Get config for Decoder
    config = BertConfig.from_pretrained(bert_model_type)
    config.output_hidden_states = True

    if model_dir is not None:
        bert = BertModel.from_pretrained(model_dir, config=config)
    else:
        bert = BertModel.from_pretrained(bert_model_type, config=config)

    if decoder_path is not None:
        decoder = HuggingDecoder.load_from_checkpoint(
            decoder_path, bert=bert, config=config)
    else:
        hparams = DotMap()
        hparams.bert_model_type = bert_model_type
        decoder = HuggingDecoder(hparams=hparams, bert=bert, config=config)
    return decoder


if __name__ == "__main__":
    main()
