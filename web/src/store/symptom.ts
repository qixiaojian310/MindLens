import { defineStore } from 'pinia'
import { v4 as uuidv4 } from 'uuid'
import type { History, HistoryItem, SymptomState } from './stateType'

export const useSymptomStore = defineStore('symptom', {
  state: (): SymptomState => ({ symptom: [], histories: [], index: 0, newHistory: [] }),
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
    initIndex(){
      this.index = 0
    },
    addHistories(history: History, name: string){
      const key = uuidv4()
      this.histories.push({ history, name, key })
    },
    deleteHistories(key: string){
      this.histories = this.histories.filter((history) => {
        return history.key !== key
      })
    },
    clearHistories(){
      this.histories = []
    },
    changeHistory(newItem: HistoryItem){
      this.newHistory[this.index] = newItem
    },
    initNewHistory(){
      this.newHistory = this.symptom.map((symptom) => {
        return {
          name: symptom,
          value: false,
        }
      })
    },
  },
})
