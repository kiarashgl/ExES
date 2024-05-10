<template>
  <v-app>
    <v-app-bar color="primary">
      <v-app-bar-title>ExES: Explaining Expert Search and Team Formation systems</v-app-bar-title>
      <v-spacer></v-spacer>

      <v-menu>
        <template v-slot:activator="{ props }">
          <v-btn color="white" v-bind="props">Toy Examples</v-btn>
        </template>

        <v-list :items="toyExamples">
          <v-list-item link
                       v-for="(item, index) in toyExamples"
                       :key="index"
          >
            <v-list-item-title @click="runToyExample(item)">{{ item.title }}</v-list-item-title>
          </v-list-item>
        </v-list>
      </v-menu>
    </v-app-bar>
    <v-main>
      <v-container fluid class="px-12">
        <v-row class="py-5 align-center d-flex">
          <v-col cols="6">

            <v-text-field hide-details color="primary" clearable variant="underlined" :placeholder="'Search ' + examples"
                          v-model="searchQuery" append-icon="mdi-magnify"
                          @keyup.enter="search" @click:append="search" :loading="searching"></v-text-field>
          </v-col>
          <v-col md="2" cols="3" style="width: 3rem">
            <v-select hide-details density="compact" v-model="dataset"
                      label="Dataset" :items="['DBLP', 'Github']"
                      variant="outlined"></v-select>
          </v-col>
          <v-col class="d-inline-flex align-center " md="3" cols="3">
            <v-select hide-details density="compact" v-model="model"
                      label="Model" :items="models" item-title="name" item-value="id"
                      variant="outlined"></v-select>
            <v-btn  @click="helpDialog = true" size="x-small" icon variant="text"><v-icon>mdi-help-circle</v-icon></v-btn>

            <v-dialog width="720" v-model="helpDialog"
            >
              <v-card>
                <v-card-title>
                  <div class="d-flex">
                    Model Selection
                    <v-spacer></v-spacer>
                    <v-btn size="small" icon variant="text" @click="helpDialog = false"><v-icon>mdi-close</v-icon></v-btn>
                  </div>

                </v-card-title>
                <v-card-text>
                  <p>
                    You may select the underlying model used in the expert search system from the combo-box.
                    </p>
                  <p>
                    The model titles (GraphSAGE and GCN) represent the graph convolutional layers used in the expert search model.
                    </p>
                  <p>
                    Furthermore, the numbers written in parentheses represent the dimensions of the hidden layers. For instance,
                    GCN (256 - 64) corresponds to an expert search system that employs two GCN (Graph Convolutional Network) layers,
                    each having 256 and 64-dimensional embeddings, respectively.
                  </p>
                </v-card-text>
              </v-card>
            </v-dialog>
          </v-col>
          <v-col md="1" cols="2">
            <v-text-field type="number" hide-details density="compact" v-model.number="topk"
                          label="Top-k" min="10" max="30"
                          variant="outlined"></v-text-field>
          </v-col>
        </v-row>
        <div v-if="results.length > 0" class="mt-1">
          <h2>Expert Search</h2>
          <v-row class="my-1">
            <v-col md="3" cols="12">
              <v-card title="Retrieved Experts">
                <v-card-text>
                  <span class="system-hint">Click on any expert in the top-k table to explain their ranking.</span>
                  <Ranking :selected="selected" :results="results" :showSkills="showSkills" :topk="topk"
                           @select-row="setSelected"></Ranking>
                  <div class="mt-2 system-hint">To explain the ranking of an expert who is out of top-k, select them
                    from the combo-box below.
                  </div>
                  <v-autocomplete class="mt-4" hide-details variant="outlined" label="Find expert" :items="results"
                                  return-object
                                  :item-title="(item) => `${item.rank + 1} - ${item.name}`"
                                  v-model="selected" @update:modelValue="setSelected(selected)"></v-autocomplete>
                </v-card-text>
              </v-card>
            </v-col>

            <v-col>
              <Explainer ref="explainer" :selected="explainerMember" :topk="explanationTopk" :dataset="setDataset" :model="setModel"
                         :query-emb="queryEmb" :query-words="queryWords"
                         v-if="selected != null">
              </Explainer>
            </v-col>
          </v-row>
          <v-row v-if="selected">
            <v-divider class="my-4"/>

            <v-col cols="12">
              <h2>
                Team Formation
              </h2>
              <v-btn v-if="!teamExplanation" @click="runTeamSearch">Build Team around {{selected.name}}</v-btn>

              <v-card v-else class="mt-4" id="teamExplanation">
                <v-card-title>Team Structure</v-card-title>

                <v-card-text v-if="!teamSearchPending">
                  <div class="system-hint">To start, click on each team member (displayed in blue color).
                    Then, select your desired explanation method from the Explain Results pane below.
                  </div>
                  <v-row>
                    <v-col cols="12">
                      <CollaborationGraph @upd="(item) => handleClick(item)" :graph="teamGraph"></CollaborationGraph>
                    </v-col>
                  </v-row>

                </v-card-text>
                <v-card-text v-else>
                  <v-progress-circular class="d-block" indeterminate></v-progress-circular>
                </v-card-text>
              </v-card>

                <div v-if="selectedTeamMember !== null">
                  <Explainer v-if="teamExplanationTopK > 0" class="mt-2" :selected="selectedTeamMember" :topk="teamExplanationTopK" :dataset="setDataset" :model="setModel"
                              :query-emb="queryEmb" :query-words="queryWords">
                  </Explainer>
                  <v-alert v-else type="warning">
                    No alternative members are available to replace {{selectedTeamMember.name}}.
                  </v-alert>
                </div>
            </v-col>

          </v-row>
        </div>
      </v-container>

    </v-main>
  </v-app>
</template>

<style>
#shap g svg {
  overflow: hidden !important;
}

#shap > div {
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
}


#shap > div > div:first-child {
  display: none;
}

#shap > div > div {
  margin: 0.25rem;
  /*margin-right: 0.25rem;*/
}

#shap > div > div > div:nth-child(2) {
  padding: 0.25rem !important;
}

#teamExplanation #teamTable .v-input--density-compact .v-field__input {
  --v-field-input-padding-top: 2px !important;
  --v-field-input-padding-bottom: 2px !important;;
  --v-input-control-height: 20px !important;
  /*font-size:.875rem!important*/
  min-height: 30px !important;
}

#teamExplanation .v-select__selection, #teamExplanation .v-input .v-label:not(.v-field-label--floating) {
  font-size: .875rem !important
}

table {
  border-collapse: collapse;
}

.system-hint {
  color: grey;
}
</style>

<script>
import axios from "axios";
import Ranking from "@/components/Ranking.vue";
import Explanation from "@/components/Explanation.vue";
import CollaborationGraph from "@/components/CollaborationGraph.vue";
import Explainer from "@/components/Explainer.vue";

export default {
  name: 'App',

  components: {
    CollaborationGraph,
    Explainer,
    Ranking
  },
  data: () => ({
    toyExamples: [
      {
        title: "Use Case 1: Influential Skills (Saliency)",
        desiredExpert: 10,
        query: "social graph",
        topk: 10,
        dataset: "DBLP",
        explanation: 6,
        tab: 1,
        radius: 0,
      },
      {
        title: "Use Case 2: Influential Edges (Saliency)",
        desiredExpert: 10,
        query: "social graph",
        topk: 10,
        dataset: "DBLP",
        explanation: 8,
        tab: 1,
        radius: 2,
      },
      {
        title: "Use Case 3: Counterfactual Skill Explanation",
        desiredExpert: 11,
        query: "database management quality",
        topk: 10,
        dataset: "DBLP",
        explanation: 0,
        tab: 2,
        radius: 1
      },
      {
        title: "Use Case 4: Counterfactual Collaboration Explanation",
        desiredExpert: 11,
        query: "database management quality",
        topk: 10,
        dataset: "DBLP",
        explanation: 4,
        tab: 2,
        radius: 1,
      },
    ],
    model: 1,
    appMode: "expert",
    searchQuery: "",
    selectedId: 0,
    selected: null,
    showSkills: false,
    topk: 10,
    radius: 1,
    explanationTopk: 10,
    dataset: "DBLP",
    queryWords: [],
    setDataset: "",
    setModel: 1,
    queryEmb: [],
    results: {},
    searching: false,
    newTopK: null,
    newRank: null,
    expert: null,
    teamExplanation: null,
    teamExplanationTopK: 0,
    selectedTeamMember: null,
    teamSearchPending: false,
    explainerMember: undefined,
    teamGraph: undefined,
    helpDialog: false,
  }),
  computed: {
    examples: function () {
      return this.dataset === "DBLP" ? '(e.g., "social graph", "database management quality")' : '(e.g., "ios swift")'
    },
    models: function() {
      return this.dataset === "DBLP" ? [{"name": "GraphSAGE (128 - 64)", id: 1}, {"name": "GCN (256 - 64)", id: 2}]
        : [{"name": "GraphSAGE (128 - 64)", id: 1}, {"name": "GCN (256 - 64)", id: 2}];
    }
  },
  methods: {
    search: async function () {
      this.searching = true
      const url = this.appMode === "expert" ? "search" : "search"
      const res = await axios.get(`http://localhost:8094/${url}`, {
        params: {
          query: this.searchQuery,
          n: this.topk,
          dataset: this.dataset,
          model: this.model
        }
      })
      const data = res.data
      this.results = data.answer
      this.queryWords = data.query_words
      this.queryEmb = data.query_embedding
      this.searching = false
      this.setDataset = this.dataset
      this.setModel = this.model
    },

    teamSearch: async function () {
      const res = await axios.post(`http://localhost:8090/teamsearch`,
        {
          query: this.queryWords, qemb: this.queryEmb, expert: this.selected.id, topk: this.topk, dataset: this.dataset, model: this.model
        })

      const data = res.data
      this.teamExplanation = data.team
      this.teamGraph = data.graph
    },

    setSelected: async function (e) {
      this.selected = e;
      this.explainerMember = e;
      this.explanations = [];
      this.explanation = null;
      this.plot = null;
      this.explanationTopk = this.topk
      this.teamExplanation = null
      this.selectedTeamMember = null
    },
    runTeamSearch: async function () {
      this.teamSearchPending = true
      await this.teamSearch()
      this.teamSearchPending = false
    },
    handleClick: function(item) {
      const nodeId = +item[0].replace("node", "")
      let maxRank = 0;
      if (item.length > 0) {
        for (const keyword in this.teamExplanation) {
          if (this.teamExplanation[keyword].selected.id === nodeId)
            if (this.teamExplanation[keyword].candidates.length > 1)
              maxRank = Math.max(maxRank, this.teamExplanation[keyword].candidates[1].rank)
        }
        const selectedPerson = this.results.filter(item => item.id === nodeId)
        this.selectedTeamMember = selectedPerson[0]
        this.teamExplanationTopK = maxRank
      }
      else this.selectedTeamMember = null
    },
    runToyExample: async function (item) {
      this.searchQuery = item.query;
      this.topk = item.topk;
      this.dataset = item.dataset;
      this.radius = item.radius;
      await this.search();
      await this.setSelected(this.results[item.desiredExpert - 1])
      this.$refs.explainer.runExample(item)
    }
  }
};
</script>
