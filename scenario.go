package novelai

// Scenario represents a NovelAI scenario JSON file (version 3, lorebook version 6).
type Scenario struct {
	ScenarioVersion      int              `json:"scenarioVersion"`
	Title                string           `json:"title"`
	Author               string           `json:"author,omitempty"`
	Description          string           `json:"description"`
	Prompt               string           `json:"prompt"`
	Tags                 []string         `json:"tags,omitempty"`
	Context              []ContextEntry   `json:"context,omitempty"`
	EphemeralContext     []ContextEntry   `json:"ephemeralContext,omitempty"`
	Settings             ScenarioSettings `json:"settings,omitempty"`
	Lorebook             Lorebook         `json:"lorebook,omitempty"`
	Placeholders         []Placeholder    `json:"placeholders,omitempty"`
	StoryContextConfig   *ContextConfig   `json:"storyContextConfig,omitempty"`
	ContextDefaults      *ContextDefaults `json:"contextDefaults,omitempty"`
	PhraseBiasGroups     []BiasGroup      `json:"phraseBiasGroups,omitempty"`
	BannedSequenceGroups []BiasGroup      `json:"bannedSequenceGroups,omitempty"`
	MessageSettings      *MessageSettings `json:"messageSettings,omitempty"`
	UserScripts          []any            `json:"userScripts,omitempty"`
}

// ContextConfig controls how a context entry is inserted into the final prompt.
type ContextConfig struct {
	Prefix               string `json:"prefix,omitempty"`
	Suffix               string `json:"suffix,omitempty"`
	TokenBudget          int    `json:"tokenBudget,omitempty"`
	ReservedTokens       int    `json:"reservedTokens,omitempty"`
	BudgetPriority       int    `json:"budgetPriority,omitempty"`
	TrimDirection        string `json:"trimDirection,omitempty"`
	InsertionType        string `json:"insertionType,omitempty"`
	MaximumTrimType      string `json:"maximumTrimType,omitempty"`
	InsertionPosition    int    `json:"insertionPosition,omitempty"`
	AllowInnerInsertion  bool   `json:"allowInnerInsertion,omitempty"`
	AllowInsertionInside bool   `json:"allowInsertionInside,omitempty"`
	Forced               bool   `json:"forced,omitempty"`
}

// ContextEntry is a piece of text with insertion rules.
type ContextEntry struct {
	Text       string         `json:"text,omitempty"`
	ContextCfg *ContextConfig `json:"contextConfig,omitempty"`
}

// ContextDefaults holds default configs for ephemeral and lore entries.
type ContextDefaults struct {
	EphemeralDefaults []ContextEntry `json:"ephemeralDefaults,omitempty"`
	LoreDefaults      []ContextEntry `json:"loreDefaults,omitempty"`
}

// Lorebook contains world info / lorebook entries.
type Lorebook struct {
	Version    int              `json:"lorebookVersion"`
	Entries    []LorebookEntry  `json:"entries"`
	Settings   LorebookSettings `json:"settings"`
	Categories []Category       `json:"categories"`
	Order      []string         `json:"order,omitempty"`
}

// LorebookSettings configures lorebook behavior.
type LorebookSettings struct {
	OrderByKeyLocations bool `json:"orderByKeyLocations"`
}

// LorebookEntry is a single lorebook entry.
type LorebookEntry struct {
	Text                string         `json:"text,omitempty"`
	ContextCfg          *ContextConfig `json:"contextConfig,omitempty"`
	LastUpdatedAt       int64          `json:"lastUpdatedAt,omitempty"`
	DisplayName         string         `json:"displayName,omitempty"`
	ID                  string         `json:"id,omitempty"`
	Keys                []string       `json:"keys,omitempty"`
	SearchRange         int            `json:"searchRange,omitempty"`
	Enabled             bool           `json:"enabled"`
	ForceActivation     bool           `json:"forceActivation,omitempty"`
	KeyRelative         bool           `json:"keyRelative,omitempty"`
	NonStoryActivatable bool           `json:"nonStoryActivatable,omitempty"`
	Category            string         `json:"category,omitempty"`
	LoreBiasGroups      []BiasGroup    `json:"loreBiasGroups,omitempty"`
	AdvancedConditions  []any          `json:"advancedConditions,omitempty"`
}

// Category is a lorebook category.
type Category struct {
	Name                string         `json:"name,omitempty"`
	ID                  string         `json:"id,omitempty"`
	Enabled             bool           `json:"enabled,omitempty"`
	CreateSubcontext    bool           `json:"createSubcontext,omitempty"`
	SubcontextSettings  *LorebookEntry `json:"subcontextSettings,omitempty"`
	UseCategoryDefaults bool           `json:"useCategoryDefaults,omitempty"`
	CategoryDefaults    *LorebookEntry `json:"categoryDefaults,omitempty"`
	CategoryBiasGroups  []BiasGroup    `json:"categoryBiasGroups,omitempty"`
	Settings            map[string]any `json:"settings,omitempty"`
	Order               []string       `json:"order,omitempty"`
	Open                bool           `json:"open,omitempty"`
}

// ScenarioSettings contains generation parameters and model config.
type ScenarioSettings struct {
	Parameters          *GenerationParams `json:"parameters,omitempty"`
	Preset              string            `json:"preset,omitempty"`
	TrimResponses       bool              `json:"trimResponses,omitempty"`
	BanBrackets         bool              `json:"banBrackets,omitempty"`
	DefaultBias         bool              `json:"defaultBias,omitempty"`
	Prefix              string            `json:"prefix,omitempty"`
	DynamicPenaltyRange bool              `json:"dynamicPenaltyRange,omitempty"`
	PrefixMode          int               `json:"prefixMode,omitempty"`
	Mode                int               `json:"mode,omitempty"`
	Model               string            `json:"model,omitempty"`
}

// GenerationParams contains sampler settings.
type GenerationParams struct {
	TextGenerationSettingsVersion     int            `json:"textGenerationSettingsVersion"`
	Temperature                       float64        `json:"temperature"`
	MaxLength                         int            `json:"max_length"`
	MinLength                         int            `json:"min_length"`
	TopK                              int            `json:"top_k"`
	TopP                              float64        `json:"top_p"`
	TopA                              float64        `json:"top_a"`
	TypicalP                          float64        `json:"typical_p"`
	TailFreeSampling                  float64        `json:"tail_free_sampling"`
	RepetitionPenalty                 float64        `json:"repetition_penalty"`
	RepetitionPenaltyRange            int            `json:"repetition_penalty_range"`
	RepetitionPenaltySlope            float64        `json:"repetition_penalty_slope"`
	RepetitionPenaltyFrequency        float64        `json:"repetition_penalty_frequency"`
	RepetitionPenaltyPresence         float64        `json:"repetition_penalty_presence"`
	RepetitionPenaltyDefaultWhitelist bool           `json:"repetition_penalty_default_whitelist"`
	CFGScale                          float64        `json:"cfg_scale"`
	CFGUC                             string         `json:"cfg_uc"`
	PhraseRepPen                      string         `json:"phrase_rep_pen"`
	TopG                              int            `json:"top_g"`
	MirostatTau                       float64        `json:"mirostat_tau"`
	MirostatLR                        float64        `json:"mirostat_lr"`
	Math1Temp                         float64        `json:"math1_temp"`
	Math1Quad                         float64        `json:"math1_quad"`
	Math1QuadEntropyScale             float64        `json:"math1_quad_entropy_scale"`
	MinP                              float64        `json:"min_p"`
	Order                             []SamplerOrder `json:"order"`
}

// SamplerOrder defines sampler ordering.
type SamplerOrder struct {
	ID      string `json:"id"`
	Enabled bool   `json:"enabled"`
}

// Placeholder is a user-configurable variable.
type Placeholder struct {
	Key             string `json:"key"`
	Description     string `json:"description"`
	DefaultValue    string `json:"defaultValue"`
	LongDescription string `json:"longDescription,omitempty"`
}

// BiasGroup configures token probability biases.
type BiasGroup struct {
	Phrases              []string `json:"phrases"`
	EnsureSequenceFinish bool     `json:"ensureSequenceFinish"`
	GenerateOnce         bool     `json:"generateOnce"`
	Bias                 float64  `json:"bias"`
	Enabled              bool     `json:"enabled"`
	WhenInactive         bool     `json:"whenInactive"`
}

// MessageSettings configures system prompt and prefill for chat models (GLM-4).
type MessageSettings struct {
	SystemPrompt string `json:"systemPrompt"`
	Prefill      string `json:"prefill"`
}

// NewScenario creates a new scenario with sensible defaults for GLM-4.
func NewScenario(title string) *Scenario {
	return &Scenario{
		ScenarioVersion:  3,
		Title:            title,
		Description:      "",
		Tags:             []string{},
		Context:          []ContextEntry{},
		EphemeralContext: []ContextEntry{},
		Placeholders:     []Placeholder{},
		Settings: ScenarioSettings{
			Preset:        "default-glm",
			TrimResponses: true,
			BanBrackets:   true,
			Prefix:        "vanilla",
			PrefixMode:    0,
			Mode:          1, // Adventure mode
			Model:         "glm-4-6",
			Parameters:    DefaultGenerationParams(),
		},
		Lorebook: Lorebook{
			Version:    6,
			Entries:    []LorebookEntry{},
			Categories: []Category{},
			Settings:   LorebookSettings{OrderByKeyLocations: false},
		},
		PhraseBiasGroups:     []BiasGroup{},
		BannedSequenceGroups: []BiasGroup{},
		UserScripts:          []any{},
	}
}

// DefaultGenerationParams returns sensible defaults for GLM-4.
func DefaultGenerationParams() *GenerationParams {
	return &GenerationParams{
		TextGenerationSettingsVersion: 8,
		Temperature:                   1,
		MaxLength:                     256,
		MinLength:                     1,
		TopK:                          40,
		TopP:                          0.95,
		TopA:                          1,
		TypicalP:                      1,
		TailFreeSampling:              1,
		PhraseRepPen:                  "medium",
		MirostatLR:                    1,
		CFGScale:                      1,
		Order: []SamplerOrder{
			{ID: "temperature", Enabled: true},
			{ID: "top_k", Enabled: true},
			{ID: "top_p", Enabled: true},
			{ID: "min_p", Enabled: false},
		},
	}
}

// DefaultContextConfig returns standard context config for lorebook entries.
func DefaultContextConfig() *ContextConfig {
	return &ContextConfig{
		Prefix:            "",
		Suffix:            "\n",
		TokenBudget:       1,
		ReservedTokens:    0,
		BudgetPriority:    400,
		TrimDirection:     "trimBottom",
		InsertionType:     "newline",
		MaximumTrimType:   "sentence",
		InsertionPosition: -1,
	}
}

// MemoryContextConfig returns context config for Memory (context[0]).
func MemoryContextConfig() *ContextConfig {
	return &ContextConfig{
		Prefix:            "",
		Suffix:            "\n",
		TokenBudget:       1,
		ReservedTokens:    0,
		BudgetPriority:    800,
		TrimDirection:     "trimBottom",
		InsertionType:     "newline",
		MaximumTrimType:   "sentence",
		InsertionPosition: 0,
	}
}

// AuthorsNoteContextConfig returns context config for Author's Note (context[1]).
func AuthorsNoteContextConfig() *ContextConfig {
	return &ContextConfig{
		Prefix:            "",
		Suffix:            "\n",
		TokenBudget:       1,
		ReservedTokens:    1,
		BudgetPriority:    -400,
		TrimDirection:     "trimBottom",
		InsertionType:     "newline",
		MaximumTrimType:   "sentence",
		InsertionPosition: -4,
	}
}

// DefaultBiasGroup returns a default (inactive) bias group.
func DefaultBiasGroup() BiasGroup {
	return BiasGroup{
		Phrases:              []string{},
		EnsureSequenceFinish: false,
		GenerateOnce:         true,
		Bias:                 0,
		Enabled:              true,
		WhenInactive:         false,
	}
}
