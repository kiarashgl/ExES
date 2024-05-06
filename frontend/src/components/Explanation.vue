<template>
  <div v-if="exp">

    <div>
      <h2>Explanation</h2>
    </div>
    <v-row>
      <v-col cols="6" xl="4">
        <h3>Ranking after perturbation</h3>
        <Ranking :results="exp.new_tops" :showSkills="false" :topk="topk" :newRank="{index: exp.new_rank, expert: expert}"></Ranking>
      </v-col>
      <v-col md="6" v-if="exp.new_graph">
        <CollaborationGraph :graph="exp.new_graph"></CollaborationGraph>
      </v-col>
    </v-row>
  </div>


</template>

<script>
import Ranking from "@/components/Ranking.vue";
import CollaborationGraph from "@/components/CollaborationGraph.vue";

export default {
  name: "Explanation",
  components: {
    CollaborationGraph,
    Ranking
  },
  props: {
    topk: Number,
    exp: Object,
    expert: Object,
  },

}
</script>

<style scoped>

</style>
