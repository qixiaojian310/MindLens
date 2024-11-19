import { defineStore } from 'pinia'

interface SymptomState {
  symptom: string[]
  index: number
}

export const useSymptomStore = defineStore('symptom', {
  state: (): SymptomState => ({ symptom: [], index: 0 }),
  actions: {
    setSymptom(symptom: string[]) {
      this.symptom = symptom
    },
    increment() {
      this.index++
    },
    decrement() {
      this.index--
    },
  },
})
