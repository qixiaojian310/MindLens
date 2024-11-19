import { defineStore } from 'pinia'

interface PredictionResult{
  model: 'cnn' | 'xgb'
  result: {
    BipolarDisorder: number
    Depression: number
    AnxietyDisorder: number
    Schizophrenia: number
    PTSD: number
  }
}

interface PredictionState{
  results: PredictionResult[]
}
export const usePredictionResultStore = defineStore('predictionResult', {
  state: (): PredictionState => ({ results: [] }),
  actions: {
    setPredictionResults(results: PredictionResult[]){
      this.results = results
    },
  },
})
