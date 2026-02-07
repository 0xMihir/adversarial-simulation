use rust_bert::pipelines::zero_shot_classification::ZeroShotClassificationModel;

pub struct ClassificationPipeline {
    model: ZeroShotClassificationModel
}

#[derive(Debug, Clone)]
pub struct ClassificationConfig {

}

// impl Default for ClassificationConfig {
//     fn default() -> Self {
//
//     }
// }

impl ClassificationPipeline {
    pub fn new(config: ClassificationConfig) -> Self {
        self.model = ZeroShotClassificationModel
    }
    
}



