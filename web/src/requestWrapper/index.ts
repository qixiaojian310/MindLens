import { useSymptomStore } from '~/store/symptom'

const baseReqURL = import.meta.env.VITE_BACK_BASE_URL

interface Symptom {
  name: string
  value: boolean
}

export interface PredictionReq {
  features: Symptom[]
  worker_choice: 'cnn' | 'xgb'
}

export async function requestSymptom() {
  const res = await fetch(`${baseReqURL}/symptom`, {
    method: 'GET',
  })
  return await res.json()
}

export async function requestPrediction(req: PredictionReq) {
  const res = await fetch(`${baseReqURL}/prediction`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json', // 添加 Content-Type 请求头
    },
    body: JSON.stringify(req),
  })
  return await res.json()
}
