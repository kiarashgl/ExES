<template>
  <v-row>

    <v-col>
      <v-card>
        <v-card-title>Explain Results
        <span class="text-subtitle-2">for {{selected.name}}</span>
        </v-card-title>
        <v-card-text>
          <div class="system-hint">Select an explanation category from tabs below.</div>
          <v-tabs v-model="tab">
            <v-tab :value="1">Saliency</v-tab>
            <v-tab :value="2">Counterfactual</v-tab>
          </v-tabs>
          <div class="my-4 d-flex">
              <v-select style="max-width: 400px" hide-details density="compact" return-object v-model="explanationMethod"
                        label="Explanation method" :items="availableExplanations" item-title="title" item-value="type"
                        variant="outlined" @change="explanation = null; plot = null"></v-select>

              <v-text-field style="max-width:150px" v-model.number="radius" variant="outlined" class="mx-2" label="Neighbourhood Radius" density="compact" hide-details type="number" min="0" max="1"></v-text-field>
              <v-text-field v-if="tab === 2" style="max-width:150px" v-model.number="numExplanations" variant="outlined" class="mx-2" label="# of Explanations" density="compact" hide-details type="number" min="1" max="10"></v-text-field>
              <v-text-field v-if="tab === 2" style="max-width:150px" v-model.number="maxExplanationSize" variant="outlined" class="mx-2" label="Max Explanation Size" density="compact" hide-details type="number" min="1" max="10"></v-text-field>
              <v-btn class="ml-3" :disabled="loading" @click="runExplanation">Explain</v-btn>
          </div>
          <v-window v-model="tab">
            <v-window-item transition="tab-transition" reverse-transition="tab-transition" :value="1">
              <div>
                <v-progress-circular class="d-block" v-if="loading" indeterminate></v-progress-circular>
                <CollaborationGraph shap v-if="shapgraph" :graph="shapgraph"></CollaborationGraph>
                <div id="shap" v-if="plot.html && !shapgraph" v-html="plot.html" :class="{flip: plot.method === 'rank'}"></div>

              </div>
            </v-window-item>
            <v-window-item transition="tab-transition" reverse-transition="tab-transition" :value="2">
              <v-progress-circular  class="d-block" v-if="loading" indeterminate></v-progress-circular>

              <v-row>
              <v-col md="4">
                <div v-if="explanations.length">
                  <v-table hover density="compact">
                    <thead>
                    <tr>
                      <th style="width: 8%">#</th>
                      <th>Explanation</th>
                      <th style="width: 25%">New Rank</th>
                    </tr>
                    </thead>
                    <tbody>
                    <tr v-for="(exp, index) in explanations" :key="index" @click="explanation = exp"
                        :style="[explanation && explanation === exp ? {'background-color': '#e7f6ff'} : {}]">
                      <td>{{ index + 1 }}</td>
                      <td>
                        <v-chip size="small" class="mr-1 my-1" v-for="item in exp.changes" :key="item"> {{ item }}</v-chip>
                      </td>
                      <td>{{ exp.new_rank }}</td>
                    </tr>
                    </tbody>
                  </v-table>
                </div>

                <div v-else-if="error">
                  <v-alert type="error">
                    No explanations were found
                  </v-alert>
                </div>
              </v-col>
              <v-col>
                <Explanation :exp="explanation" :expert="selected" :topk="topk"></Explanation>
              </v-col>


              </v-row>
            </v-window-item>
          </v-window>

        </v-card-text>
      </v-card>

    </v-col>
  </v-row>
</template>

<script>
import axios from "axios";
import CollaborationGraph from "@/components/CollaborationGraph.vue";
import Explanation from "@/components/Explanation.vue";

export default {
  name: "Explainer",
  components: {
    CollaborationGraph,
    Explanation,
  },
  props: {
    selected: Object,
    topk: Number,
    queryEmb: Array,
    queryWords: Array,
    dataset: String,
    model: Number
  },
  watch: {
    selected: {
      handler: 'reset'
    }
  },
  computed: {
    availableExplanations() {
      if (this.selected === null)
        return null
      if (this.tab === 1)
        return this.explanationMethods.filter(item => item.mode === "saliency")
      const inTopK = this.selected.rank < this.topk;
      return this.explanationMethods.filter(item => item.mode === "counterfactual" && item.topk === inTopK)
    }
  },
  data: function () {
    return {
      tab: null,
      plot: "",
      shapgraph: null,
      explanationMethod: null,
      radius:1,
      numExplanations:5,
      maxExplanationSize:5,
      explanationMethods: [
        {
          title: "Add skill to expert",
          type: "add_skill_expert",
          mode: "counterfactual",
          topk: false,
          hint: "Add these skills to selected expert"
        },
        {
          title: "Remove skill from expert",
          type: "remove_skill_expert",
          mode: "counterfactual",
          topk: true,
          hint: "Remove these skills from selected expert"
        },
        {
          title: "Add keyword to query",
          type: "add_skill_query",
          mode: "counterfactual",
          topk: false,
          hint: "Add these skills to query"
        },
        {
          title: "Add keyword to query",
          type: "add_skill_query",
          mode: "counterfactual",
          topk: true,
          hint: "Add these skills to query"
        },
        {
          title: "Add collaboration",
          type: "add_edge_expert",
          mode: "counterfactual",
          topk: false,
          hint: "Add these edges to selected expert"
        },
        {
          title: "Remove collaboration",
          type: "remove_edge_expert",
          mode: "counterfactual",
          topk: true,
          hint: "Remove these edges from selected expert"
        },
        {title: "Shap values for skills", type: "shap", mode: "saliency", hint: "Shap values for skills"},
        {title: "Shap values for query", type: "shap_query", mode: "saliency", hint: "Shap values for query"},
        {title: "Shap values for edges", type: "shap_edge", mode: "saliency", hint: "Shap values for edges"},
      ],
      explanations: [],
      explanation: null,
      error: false,
      loading: false,
      saliency: {
        shapSkills: {}
      }
    }
  },
  methods: {
    reset: function() {
      this.saliency.shapSkills =  {}
      this.explanations = []
      this.explanation = null
      this.explanationMethod = null
      this.shapgraph = null
      this.plot = {html: ""}
    },
    getColor: function(skill) {
      if (skill in this.saliency.shapSkills)
        return `rgba(${this.saliency.shapSkills[skill]})`
      else return undefined
    },
    runExplanation: async function () {
      this.error = false
      this.explanations = [];
      this.explanation = null;
      this.loading = false;
      this.plot = {html: ""};
      try {
        this.loading = true;

        const res = await axios.post(`http://localhost:8094/explain/${this.explanationMethod.type}`,
          {
            query: this.queryWords, qemb: this.queryEmb, expert: this.selected.id, topk: this.topk, num_explanations: this.numExplanations,
            max_explanation_size: this.maxExplanationSize,
            expert_in_topk: this.selected.rank < this.topk, dataset: this.dataset, model: this.model, radius: this.radius,
          })
        const data = res.data

        if (this.explanationMethod.type === "shap" || this.explanationMethod.type === "shap_query" || this.explanationMethod.type === "shap_edge") {
          this.plot.html = data.plot
          if (this.explanationMethod.type === "shap") {
              this.saliency.shapSkills = data.colors
          }


        } else {
          this.explanations = data
        }
        if (this.explanationMethod.type === "shap_edge")
          this.shapgraph = {nodes: data.nodes, edges: data.edges}
        else
          this.shapgraph = null
        this.expert = this.selected
      } catch (err) {
        this.error = true
      }
      this.loading = false;
    },
    runExample: function(item) {
      this.tab = item.tab
      this.radius = item.radius
      this.explanationMethod = this.explanationMethods[item.explanation]
      this.numExplanations = 5;
      this.maxExplanationSize = 5;
      this.runExplanation()
    }
  }
}
</script>

<style>
#shap>div {
  display: none !important;
}

#shap.flip {
  transform: scale(-1, 1)!important;
}
#shap.flip text {
  transform: scale(-1, 1) !important;
}
</style>
