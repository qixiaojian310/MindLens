import { defineStore } from 'pinia'
import type { PredictionResult } from '~/types/result'

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
