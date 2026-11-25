param location string
param storageAccountName string
param registryName string
param imageDigest string
param operatorPrincipalId string
param agentPrincipalId string

var roleApi = 'Microsoft.Authorization/roleDefinitions'
var blobReader = subscriptionResourceId(roleApi, '2a2b9908-6ea1-4ae2-8e65-a410df84e7d1')
var blobWriter = subscriptionResourceId(roleApi, 'ba92f5b4-2d11-453d-a403-e96b0029c9fe')
var queueContributor = subscriptionResourceId(roleApi, '974c5e8b-45b9-4653-ba55-5f855dd0fb88')
var queueSender = subscriptionResourceId(roleApi, 'c6a89b2d-59bc-44d0-9896-0f6e12d7b80a')
var acrPull = subscriptionResourceId(roleApi, '7f951dda-4ed3-4680-a7ca-43fe172d538d')

resource storage 'Microsoft.Storage/storageAccounts@2023-05-01' existing = {
  name: storageAccountName
}

resource blobs 'Microsoft.Storage/storageAccounts/blobServices@2023-05-01' existing = {
  parent: storage
  name: 'default'
}

resource plans 'Microsoft.Storage/storageAccounts/blobServices/containers@2023-05-01' existing = {
  parent: blobs
  name: 'pbrp-plans'
}

resource results 'Microsoft.Storage/storageAccounts/blobServices/containers@2023-05-01' = {
  parent: blobs
  name: 'pbrp-smoke-results'
  properties: { publicAccess: 'None' }
}

resource queues 'Microsoft.Storage/storageAccounts/queueServices@2023-05-01' existing = {
  parent: storage
  name: 'default'
}

resource approvals 'Microsoft.Storage/storageAccounts/queueServices/queues@2023-05-01' = {
  parent: queues
  name: 'pbrp-smoke-approvals'
  properties: {}
}

resource registry 'Microsoft.ContainerRegistry/registries@2025-11-01' existing = {
  name: registryName
}

var image = '${registry.properties.loginServer}/pbrp-smoke@${imageDigest}'

resource worker 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' = {
  name: 'pbrp-dev-smoke-worker'
  location: location
}

resource workerReadsPlans 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  scope: plans
  name: guid(plans.id, worker.id, blobReader)
  properties: {
    principalId: worker.properties.principalId
    principalType: 'ServicePrincipal'
    roleDefinitionId: blobReader
  }
}

resource workerWritesResults 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  scope: results
  name: guid(results.id, worker.id, blobWriter)
  properties: {
    principalId: worker.properties.principalId
    principalType: 'ServicePrincipal'
    roleDefinitionId: blobWriter
  }
}

resource agentReadsResults 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  scope: results
  name: guid(results.id, agentPrincipalId, blobReader)
  properties: {
    principalId: agentPrincipalId
    principalType: 'ServicePrincipal'
    roleDefinitionId: blobReader
  }
}

resource workerProcessesQueue 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  scope: approvals
  name: guid(approvals.id, worker.id, queueContributor)
  properties: {
    principalId: worker.properties.principalId
    principalType: 'ServicePrincipal'
    roleDefinitionId: queueContributor
  }
}

resource operatorApproves 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  scope: approvals
  name: guid(approvals.id, operatorPrincipalId, queueSender)
  properties: {
    principalId: operatorPrincipalId
    principalType: 'User'
    roleDefinitionId: queueSender
  }
}

resource workerPullsImage 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  scope: registry
  name: guid(registry.id, worker.id, acrPull)
  properties: {
    principalId: worker.properties.principalId
    principalType: 'ServicePrincipal'
    roleDefinitionId: acrPull
  }
}

resource environment 'Microsoft.App/managedEnvironments@2025-07-01' = {
  name: 'pbrp-dev-smoke-env'
  location: location
  properties: {
    appLogsConfiguration: { destination: 'azure-monitor' }
    workloadProfiles: [
      { name: 'Consumption', workloadProfileType: 'Consumption' }
    ]
  }
}

resource job 'Microsoft.App/jobs@2025-07-01' = {
  name: 'pbrp-dev-smoke'
  location: location
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: { '${worker.id}': {} }
  }
  properties: {
    environmentId: environment.id
    workloadProfileName: 'Consumption'
    configuration: {
      triggerType: 'Event'
      replicaTimeout: 300
      replicaRetryLimit: 0
      registries: [
        { server: registry.properties.loginServer, identity: worker.id }
      ]
      eventTriggerConfig: {
        parallelism: 1
        replicaCompletionCount: 1
        scale: {
          minExecutions: 0
          maxExecutions: 1
          pollingInterval: 30
          rules: [
            {
              name: 'approved-plan'
              type: 'azure-queue'
              identity: worker.id
              metadata: {
                accountName: storageAccountName
                queueName: approvals.name
                queueLength: '1'
              }
            }
          ]
        }
      }
    }
    template: {
      containers: [
        {
          name: 'smoke'
          image: image
          resources: { cpu: json('0.25'), memory: '0.5Gi' }
          env: [
            { name: 'AZURE_CLIENT_ID', value: worker.properties.clientId }
            { name: 'AZURE_STORAGE_ENDPOINT', value: storage.properties.primaryEndpoints.blob }
            { name: 'AZURE_STORAGE_QUEUE_ENDPOINT', value: storage.properties.primaryEndpoints.queue }
            { name: 'PBRP_SMOKE_IMAGE', value: image }
          ]
        }
      ]
    }
  }
  dependsOn: [workerPullsImage, workerReadsPlans, workerWritesResults, workerProcessesQueue]
}

output jobId string = job.id
