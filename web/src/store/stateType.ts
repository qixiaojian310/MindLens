export interface HistoryItem {
  name: string
  value: boolean
}

export type History = HistoryItem[]
export interface HistoriesItem {
  history: History
  name: string
  key: string
}
export interface SymptomState {
  symptom: string[]
  /** 当前被操作的历史记录index */
  index: number
  histories: HistoriesItem[]
  newHistory: History
}
