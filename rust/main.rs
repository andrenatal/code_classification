use ndarray::{Array2, Axis, concatenate};
use ort::{
	Error,
    execution_providers::CUDAExecutionProvider,
	session::{Session, builder::GraphOptimizationLevel}
};
use tokenizers::{Result, Tokenizer, PaddingParams, PaddingStrategy, EncodeInput};

fn main() -> ort::Result<()> {
	let inputs = vec!["I am writing this to test the model."];

    // Initialize tracing to receive debug messages from `ort`
	tracing_subscriber::fmt::init();

	ort::init()
		.with_name("sbert")
		.with_execution_providers([CUDAExecutionProvider::default().build()])
		.commit()?;
	let session = Session::builder()?
		.with_optimization_level(GraphOptimizationLevel::Level1)?
		.with_intra_threads(1)?
		.commit_from_file("/media/4tbdrive/engines/code_classification/onnx/distilbert_base_uncased/model.onnx")?;
	let mut tokenizer = Tokenizer::from_file("/media/4tbdrive/engines/code_classification/onnx/distilbert_base_uncased/tokenizer.json").unwrap();
    //let tokenizer = Tokenizer::from_pretrained("distilbert_base_uncased", None)?;
    tokenizer.with_padding(Some(PaddingParams {
        strategy: PaddingStrategy::BatchLongest,
        ..Default::default()
    }));
    let encodings = tokenizer.encode_batch(inputs.iter().map(|s| EncodeInput::Single(s.clone().into())).collect(), true).map_err(|e| Error::new(e.to_string()))?;
	let padded_token_length = encodings[0].len();
    //println!("encodings: {}", padded_token_length);
	let ids: Vec<i64> = encodings.iter().flat_map(|e| e.get_ids().iter().map(|i| *i as i64)).collect();
	let mask: Vec<i64> = encodings.iter().flat_map(|e| e.get_attention_mask().iter().map(|i| *i as i64)).collect();
	let a_ids = Array2::from_shape_vec([inputs.len(), padded_token_length], ids).unwrap();
	let a_mask = Array2::from_shape_vec([inputs.len(), padded_token_length], mask).unwrap();
    //println!("ids: {:?}", a_ids);
    //println!("ids: {:?}", a_mask);
    let outputs = session.run(ort::inputs![a_ids, a_mask]?)?;
    let last_hidden_state = outputs["last_hidden_state"].try_extract_tensor::<f32>()?;
    let text_embs = last_hidden_state.mean_axis(Axis(1)).ok_or(Error::new("mean_axis failed"))?;
    //println!("last_hidden_state mean: {:?}", text_embs);

	let session = Session::builder()?
		.with_optimization_level(GraphOptimizationLevel::Level1)?
		.with_intra_threads(1)?
		.commit_from_file("/media/4tbdrive/engines/code_classification/onnx/codebert-base/model.onnx")?;
	let mut tokenizer = Tokenizer::from_file("/media/4tbdrive/engines/code_classification/onnx/codebert-base/tokenizer.json").unwrap();
    tokenizer.with_padding(Some(PaddingParams {
        strategy: PaddingStrategy::BatchLongest,
        ..Default::default()
    }));
    let encodings = tokenizer.encode_batch(inputs.iter().map(|s| EncodeInput::Single(s.clone().into())).collect(), true).map_err(|e| Error::new(e.to_string()))?;
    let padded_token_length = encodings[0].len();
	let ids: Vec<i64> = encodings.iter().flat_map(|e| e.get_ids().iter().map(|i| *i as i64)).collect();
	let mask: Vec<i64> = encodings.iter().flat_map(|e| e.get_attention_mask().iter().map(|i| *i as i64)).collect();
	let a_ids = Array2::from_shape_vec([inputs.len(), padded_token_length], ids).unwrap();
	let a_mask = Array2::from_shape_vec([inputs.len(), padded_token_length], mask).unwrap();
	let outputs = session.run(ort::inputs![a_ids, a_mask]?)?;
    let last_hidden_state = outputs["last_hidden_state"].try_extract_tensor::<f32>()?;
    let code_embs = last_hidden_state.mean_axis(Axis(1)).ok_or(Error::new("mean_axis failed"))?;
    //println!("last_hidden_state mean: {:?}", code_embs);

	let session = Session::builder()?
		.with_optimization_level(GraphOptimizationLevel::Level1)?
		.with_intra_threads(1)?
		.commit_from_file("/media/4tbdrive/engines/code_classification/onnx/meta_classifier.onnx")?;
    let concat_embs = concatenate(Axis(1), &[text_embs.view(), code_embs.view()]).unwrap();
    //println!("concat_embs: {:?}", concat_embs);

	let outputs = session.run(ort::inputs![concat_embs]?)?;
    //println!("outputs: {:?}", outputs);
    let logits = outputs["output"].try_extract_tensor::<f32>()?;
    //println!("logits: {:?}", logits);
    let probs = logits.mapv(|x: f32| { 1.0 / (1.0 + (-x).exp())});
    //println!("probs: {:?}", probs.iter().cloned().collect::<Vec<f32>>()[0]);
    if probs.iter().cloned().collect::<Vec<f32>>()[0] > 0.5 {
        println!("computer language, probs: {:?}", probs.iter().cloned().collect::<Vec<f32>>()[0]);
    } else {
        println!("human language, probs: {:?}", probs.iter().cloned().collect::<Vec<f32>>()[0]);
    }

	Ok(())
}