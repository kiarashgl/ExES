<template>
  <v-table density="compact" hover style="max-height: 400px;overflow: auto;">
    <thead>
    <tr>
      <th style="width: 8%">Rank</th>
      <th>Name</th>
      <th v-if="showSkills">Skills</th>
    </tr>
    </thead>
    <tbody>
    <tr v-for="(row, index) in showingResults" :key="row.id"
        :style="[newRank !== undefined && index === newRank.index - 1 ? {'background-color': '#f1e3a1'} : {},
        newRank === undefined && index === topk ? {'border-top': 'black 5px solid'} : {},
        selected && selected.id === row.id ? {'background-color': '#e7f6ff'} : {}]"
        @click="clickOnRow(row)">
      <td>{{ index + 1 }}</td>
      <td>{{ row.name }}</td>
      <td v-if="showSkills">
        <v-chip size="x-small" class="px-1 mr-1" v-for="skill in row.skills" :key="skill">{{ skill }}</v-chip>
      </td>
    </tr>
    <tr v-if="newRank !== undefined && newRank.index > topk" style="border-top: black 5px solid">
      <td>{{newRank.index}}</td>
      <td>{{newRank.expert.name}}</td>
    </tr>

    </tbody>
  </v-table>
</template>

<script>
export default {
  name: "Ranking",
  emits: ['select-row'],
  props: {
    results: Array,
    topk: Number,
    showSkills: Boolean,
    newRank: {
      type: Object,
      default: undefined
    },
    selected: Object
  },

  computed: {
    showingResults() {
      return this.results.slice(0, this.topk)
    }
  },

  data: () => ({
    // selected: null
  }),
  methods: {
    clickOnRow: function(row) {
      // this.selected = row
      this.$emit('select-row', row)
    }
  }
}
</script>

<style scoped>

</style>
