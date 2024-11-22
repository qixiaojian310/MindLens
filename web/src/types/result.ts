export interface PredictionResult{
  model: 'cnn_hierarchy' | 'cnn_random' | 'ffnn' | 'xgb'
  key: string
  name: string
  result: {
    BipolarDisorder: number
    Schizophrenia: number
    Depression: number
    AnxietyDisorder: number
    PTSD: number
  }
}
