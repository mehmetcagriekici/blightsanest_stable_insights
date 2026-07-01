package domain

/* types between api and rag */
// response received from rag
type RagResponse struct {
  ID       string `json:"id"` 
  Status   string `json:"status"`
  Response string `json:"response"`
}

// request that will be sent to rag 
type RagRequest struct {
  Query string `json:"query"`
}

/* types between api and database */
// documents
// content is compressed into string
type Document struct {
  ID      string `json:"id"`
  Content string `json:"content"`
}

/* types between api and pubsub*/
