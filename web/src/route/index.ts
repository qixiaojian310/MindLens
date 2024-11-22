import HomeView from '~/layout/HomeView.vue'
import QuestionnaireView from '~/layout/QuestionnaireView.vue'
import ResultView from '~/layout/ResultView.vue'

const routes = [
  { path: '/', component: HomeView, name: 'Home' },
  { path: '/questionnaire', component: QuestionnaireView, name: 'Questionnaire' },
  { path: '/result', component: ResultView, name: 'Result' },
]

export default routes
