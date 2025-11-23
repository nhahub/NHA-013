// ============================================
// Mental Health Chatbot - Complete System
// ============================================

// Global variables
let RESPONSES_DATA = null;
let SOLUTIONS_DATA = null;
let isConfigLoaded = false;

const API_URL = 'https://adhamelmalhy-chatbot.hf.space/predict_health';

let chatState = {
    mode: 'greeting',
    currentQuestionIndex: 0,
    collectedData: {},
    // Solutions state
    solutionsState: {
        problems: [],           // List of detected problems
        currentProblemIndex: -1, // Current problem
        currentSolutionIndex: 0, // Current solution
        currentResourceIndex: 0  // Current video/podcast
    }
};

let currentIntroStep = 0;
let isDarkMode = false;
let currentSection = 'chat';

// ============================================
// Configuration Loading Functions
// ============================================

async function loadConfigurations() {
    try {
        // Correct path: from frontend to backend/MoodMate_Backend/
        const responsesPath = '../backend/MoodMate_Backend/responses.json';
        const solutionsPath = '../backend/MoodMate_Backend/solutions.json';

        // Load responses.json
        console.log('🔍 Loading responses.json from Backend...');
        const responsesResponse = await fetch(responsesPath);
        if (!responsesResponse.ok) {
            throw new Error(`Failed to load responses.json - Status: ${responsesResponse.status}`);
        }
        RESPONSES_DATA = await responsesResponse.json();
        console.log('✅ responses.json loaded successfully');

        // Load solutions.json
        try {
            console.log('🔍 Loading solutions.json from Backend...');
            const solutionsResponse = await fetch(solutionsPath);
            if (solutionsResponse.ok) {
                SOLUTIONS_DATA = await solutionsResponse.json();
                console.log('✅ solutions.json loaded successfully');
            }
        } catch (err) {
            console.warn('⚠️ solutions.json file not found (will rely on API)');
        }

        isConfigLoaded = true;
        return true;

    } catch (error) {
        console.error('❌ Error loading files:', error);
        isConfigLoaded = false;

        const container = document.getElementById('messages-container');
        if (container) {
            container.innerHTML = `
                <div class="flex items-center justify-center h-full">
                    <div class="text-center p-8 bg-red-50 rounded-2xl max-w-md">
                        <h3 class="text-xl font-bold text-red-600 mb-3">⚠️ Loading Error</h3>
                        <p class="text-gray-700 mb-3">Failed to load configuration files. Check for:</p>
                        <ul class="text-sm text-gray-600 text-right list-disc list-inside">
                            <li>responses.json</li>
                            <li>solutions.json (optional)</li>
                        </ul>
                        <button onclick="location.reload()" class="mt-4 bg-red-600 text-white px-4 py-2 rounded-lg hover:bg-red-700">
                            Retry
                        </button>
                    </div>
                </div>
            `;
        }
        return false;
    }
}

// ============================================
// Helper Functions
// ============================================

function delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

function scrollToBottom() {
    const container = document.getElementById('messages-container');
    if (container) {
        container.scrollTop = container.scrollHeight;
    }
}

function parseMarkdown(text) {
    if (typeof text !== 'string') return '';

    // Convert bold text
    text = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');

    // Convert URLs to beautiful buttons
    const urlRegex = /(https?:\/\/[^\s]+)|(www\.[^\s]+)/g;
    text = text.replace(urlRegex, function (url) {
        const href = url.startsWith('www.') ? 'https://' + url : url;

        // Determine button text based on link type
        let buttonText = '🔗 افتح الرابط';
        let buttonColor = 'bg-blue-600 hover:bg-blue-700';

        if (url.includes('youtu')) {
            buttonText = '🎥 شاهد الفيديو';
            buttonColor = 'bg-red-600 hover:bg-red-700';
        } else if (url.includes('podcast')) {
            buttonText = '🎧 استمع للبودكاست';
            buttonColor = 'bg-purple-600 hover:bg-purple-700';
        }

        return `<a href="${href}" target="_blank" rel="noopener noreferrer" 
                   class="inline-block ${buttonColor} text-white px-6 py-3 rounded-lg font-medium shadow-lg transition-all duration-200 hover:shadow-xl hover:scale-105 my-2">
                   ${buttonText}
                </a>`;
    });

    // Convert line breaks
    text = text.replace(/\n/g, '<br>');

    return text;
}

// Translate answers to standardized keys
function getStoredKey(userMessage, questionConfig) {
    const userTextLower = userMessage.toLowerCase();
    const repliesConfig = questionConfig.answer_replies || {};

    for (const [stdKey, data] of Object.entries(repliesConfig)) {
        if (stdKey !== "Other") {
            for (const keyword of data.keywords || []) {
                if (userTextLower.includes(keyword.toLowerCase())) {
                    return { reply: data.bot_reply[0], storedKey: stdKey };
                }
            }
        }
    }

    if (repliesConfig.Other) {
        const reply = repliesConfig.Other.bot_reply[0];
        if (questionConfig.field === "Country") {
            return { reply: reply, storedKey: userMessage };
        }
        return { reply: reply, storedKey: 'Other' };
    }

    return { reply: RESPONSES_DATA.unclear_responses[0], storedKey: userMessage };
}

// ============================================
// Chat Interface Functions
// ============================================

function addUserMessage(message) {
    const messagesContainer = document.getElementById('messages-container');
    if (!messagesContainer) return;

    const messageDiv = document.createElement('div');
    messageDiv.className = 'flex justify-end message-pop';
    messageDiv.innerHTML = `
        <div class="user-message bg-blue-600 text-white rounded-2xl p-6 max-w-md shadow-xl">
            <p class="leading-relaxed text-lg">${message}</p>
        </div>
    `;
    messagesContainer.appendChild(messageDiv);
    scrollToBottom();
}

function addBotMessage(message) {
    const messagesContainer = document.getElementById('messages-container');
    if (!messagesContainer) return;

    const messageDiv = document.createElement('div');
    messageDiv.className = 'flex items-start space-x-4 message-pop';
    messageDiv.innerHTML = `
        <img src="image/download-removebg-preview.png" alt="MoodMate Avatar"
             class="w-12 h-12 rounded-full flex-shrink-0 shadow-lg object-cover">
        <div class="bot-message rounded-2xl p-6 max-w-md shadow-xl">
            <div class="prose text-gray-800 leading-relaxed text-lg page-transition">${parseMarkdown(message)}</div>
        </div>
    `;
    messagesContainer.appendChild(messageDiv);
    scrollToBottom();
}

function showTypingIndicator() {
    const messagesContainer = document.getElementById('messages-container');
    if (!messagesContainer) return;
    removeTypingIndicator();

    const typingDiv = document.createElement('div');
    typingDiv.className = 'flex items-start space-x-4';
    typingDiv.id = 'typing-indicator';
    typingDiv.innerHTML = `
        <img src="image/download-removebg-preview.png" alt="MoodMate Avatar"
            class="w-12 h-12 rounded-full flex-shrink-0 shadow-lg object-cover">
        <div class="bot-message rounded-2xl p-6 shadow-xl">
            <div class="flex space-x-2">
                <div class="w-3 h-3 bg-gray-400 rounded-full typing-dots"></div>
                <div class="w-3 h-3 bg-gray-400 rounded-full typing-dots"></div>
                <div class="w-3 h-3 bg-gray-400 rounded-full typing-dots"></div>
            </div>
        </div>
    `;
    messagesContainer.appendChild(typingDiv);
    scrollToBottom();
}

function removeTypingIndicator() {
    const typingIndicator = document.getElementById('typing-indicator');
    if (typingIndicator) typingIndicator.remove();
}

// ============================================
// API Communication and Problem Display
// ============================================

async function sendDataToAPI(collectedData) {
    showTypingIndicator();
    await delay(1500);

    try {
        const payload = {
            Gender: collectedData.Gender || 'Male',
            Country: collectedData.Country || 'Other',
            Occupation: collectedData.Occupation || 'Other',
            Growing_Stress: collectedData.Growing_Stress || 'No',
            Changes_Habits: collectedData.Changes_Habits || 'No',
            Days_Indoors: collectedData.Days_Indoors || 'Moderate',
            Mood_Swings: collectedData.Mood_Swings || 'Medium',
            Coping_Struggles: collectedData.Coping_Struggles || 'No',
            Work_Interest: collectedData.Work_Interest || 'Yes',
            Social_Weakness: collectedData.Social_Weakness || 'No',
            Mental_Health_History: collectedData.Mental_Health_History || 'No',
            family_history: collectedData.family_history || 'No',
            care_options: collectedData.care_options || 'No',
            mental_health_interview: collectedData.mental_health_interview || 'No'
        };

        console.log('📤 Sending data to API:', payload);

        const response = await fetch(API_URL, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        removeTypingIndicator();

        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`HTTP ${response.status}: ${errorText}`);
        }

        const result = await response.json();
        console.log('📥 Received result from API:', result);

        if (result.status === 'success') {
            const stability = result.stability_percentage;

            // Display stability percentage
            let predictionMessage = `بناءً على تحليل إجاباتك، نسبة **الصحة النفسية المناسبة** لديك: **${stability.toFixed(2)}%**\n\n`;
            predictionMessage += result.final_advice;

            addBotMessage(predictionMessage);
            await delay(2000);

            // Save problems and show menu
            if (result.solutions_report && result.solutions_report.problems && result.solutions_report.problems.length > 0) {
                chatState.solutionsState.problems = result.solutions_report.problems;
                await showProblemsMenu();
            } else {
                addBotMessage("✅ لم نجد أي تحديات رئيسية تحتاج لحلول فورية!");
                await delay(1500);
                await showFarewellMessages();
            }

        } else {
            addBotMessage(`❌ خطأ: ${result.detail || 'تحقق من Console'}`);
        }

    } catch (error) {
        removeTypingIndicator();
        console.error('❌ Network Error:', error);
        addBotMessage(`❌ فشل الاتصال بالخادم. الخطأ: ${error.message}\n\nتأكد من:\n• تشغيل api_server.py على المنفذ 8000\n• عدم وجود Firewall يمنع الاتصال`);
    }
}

// ============================================
// Problem and Solution Display Functions
// ============================================

async function showProblemsMenu() {
    const problems = chatState.solutionsState.problems;

    let menuMessage = "**المشاكل التي تم اكتشافها:**\n\n";
    problems.forEach((problem, index) => {
        menuMessage += `**${index + 1}.** ${problem.name}\n`;
    });
    menuMessage += "\n**اختر رقم المشكلة اللي عايز تعرف حلولها:**";

    addBotMessage(menuMessage);
    chatState.mode = 'selecting_problem';
}

async function showProblemSolutions(problemIndex) {
    const problem = chatState.solutionsState.problems[problemIndex];

    console.log('🔍 Displaying problem solutions:', problem);

    // Display problem name and description
    await delay(500);
    addBotMessage(`**${problem.name}**\n\n${problem.description}`);

    await delay(1500);
    addBotMessage("**الحلول المقترحة:**");

    // Display first solution
    if (problem.selected_solutions && problem.selected_solutions[0]) {
        await delay(1000);
        addBotMessage(problem.selected_solutions[0]);
    }

    // Display second solution
    if (problem.selected_solutions && problem.selected_solutions[1]) {
        await delay(1500);
        addBotMessage(problem.selected_solutions[1]);
    }

    // Display video (if exists)
    if (problem.video_link) {
        await delay(1500);

        // Try to get video intro from solutions.json
        let videoIntro = "**فيديو مفيد:**";
        try {
            if (SOLUTIONS_DATA && problem.key && SOLUTIONS_DATA[problem.key]) {
                videoIntro = SOLUTIONS_DATA[problem.key].video_intro || videoIntro;
            }
        } catch (err) {
            console.warn('⚠️ Failed to get video_intro:', err);
        }

        addBotMessage(videoIntro);

        await delay(1000);
        addBotMessage(problem.video_link);
    }

    // Display podcast (if exists in solutions.json)
    try {
        if (SOLUTIONS_DATA && problem.key && SOLUTIONS_DATA[problem.key] && SOLUTIONS_DATA[problem.key].podcasts && SOLUTIONS_DATA[problem.key].podcasts[0]) {
            await delay(1500);

            const podcastIntro = SOLUTIONS_DATA[problem.key].podcast_intro || "**بودكاست مفيد:**";
            addBotMessage(podcastIntro);

            await delay(1000);
            const podcast = SOLUTIONS_DATA[problem.key].podcasts[0];
            addBotMessage(podcast);
        }
    } catch (err) {
        console.warn('⚠️ Failed to get podcast:', err);
    }

    // Ask: Want to continue?
    await delay(1500);
    addBotMessage("**عايز تشوف حلول لمشكلة تانية؟** (اكتب: نعم / لا)");
    chatState.mode = 'after_problem_solutions';

    console.log('✅ All solutions displayed successfully');
}

async function showNextSolution() {
    // This function is no longer used
}

async function showResources() {
    // This function is no longer used
}

async function showFarewellMessages() {
    console.log('🔍 Starting farewell message display...');

    // Farewell message is in solutions.json under final_summary
    if (SOLUTIONS_DATA && SOLUTIONS_DATA.final_summary && SOLUTIONS_DATA.final_summary.messages) {
        console.log('✅ Farewell message found:', SOLUTIONS_DATA.final_summary.messages);

        for (const msg of SOLUTIONS_DATA.final_summary.messages) {
            await delay(1500);
            addBotMessage(msg);
        }

        console.log('✅ All farewell messages displayed');
    } else {
        console.error('❌ final_summary not found in solutions.json!');
        console.log('SOLUTIONS_DATA:', SOLUTIONS_DATA);

        // Error message for user
        addBotMessage("⚠️ حدث خطأ في تحميل رسالة النهاية. تأكد من وجود ملف solutions.json");
    }

    chatState.mode = 'finished';
}

// ============================================
// Main Initialization
// ============================================

document.addEventListener('DOMContentLoaded', async function () {

    // Load configuration files first
    const loaded = await loadConfigurations();
    if (!loaded) return;

    // Select elements
    const authSection = document.getElementById('auth-section');
    const signupForm = document.getElementById('signup-form');
    const signinForm = document.getElementById('signin-form');
    const otpModal = document.getElementById('otp-modal');
    const introSection = document.getElementById('intro-section');
    const mainInterface = document.getElementById('main-interface');
    const sidebar = document.getElementById('sidebar');
    const sidebarOverlay = document.getElementById('sidebar-overlay');
    const messageInput = document.getElementById('message-input');
    const messageForm = document.getElementById('message-form');

    // UI helper functions
    function showSection(sectionId) {
        document.querySelectorAll('#app > div').forEach(section => {
            section.classList.add('hidden');
        });
        const targetSection = document.getElementById(sectionId);
        if (targetSection) targetSection.classList.remove('hidden');
    }

    function showMainSection(sectionId) {
        document.querySelectorAll('[id$="-section"]:not(#auth-section):not(#intro-section)').forEach(section => {
            section.classList.add('hidden');
        });
        const targetSection = document.getElementById(sectionId);
        if (targetSection) targetSection.classList.remove('hidden');
        currentSection = sectionId.replace('-section', '');
    }

    function showError(errorId, formId) {
        const errorDiv = document.getElementById(errorId);
        const form = document.getElementById(formId);
        if (errorDiv) errorDiv.classList.remove('hidden');
        if (form) form.classList.add('shake');
        setTimeout(() => {
            if (form) form.classList.remove('shake');
            if (errorDiv) errorDiv.classList.add('hidden');
        }, 3000);
    }

    // Authentication handlers
    const showSigninBtn = document.getElementById('show-signin');
    if (showSigninBtn) {
        showSigninBtn.addEventListener('click', () => {
            if (signupForm) signupForm.classList.add('hidden');
            if (signinForm) signinForm.classList.remove('hidden');
            const signinEmail = document.getElementById('signin-email');
            if (signinEmail) signinEmail.focus();
        });
    }

    const showSignupBtn = document.getElementById('show-signup');
    if (showSignupBtn) {
        showSignupBtn.addEventListener('click', () => {
            if (signinForm) signinForm.classList.add('hidden');
            if (signupForm) signupForm.classList.remove('hidden');
            const signupEmail = document.getElementById('signup-email');
            if (signupEmail) signupEmail.focus();
        });
    }

    const signupSubmit = document.getElementById('signup-submit');
    if (signupSubmit) {
        signupSubmit.addEventListener('submit', function (e) {
            e.preventDefault();
            const email = document.getElementById('signup-email')?.value;
            const password = document.getElementById('signup-password')?.value;

            if (!email || !password || password.length < 6) {
                showError('signup-error', 'signup-submit');
                return;
            }
            if (otpModal) otpModal.classList.remove('hidden');
            const otpInput = document.querySelector('.otp-input');
            if (otpInput) otpInput.focus();
        });
    }

    const signinSubmit = document.getElementById('signin-submit');
    if (signinSubmit) {
        signinSubmit.addEventListener('submit', function (e) {
            e.preventDefault();
            const email = document.getElementById('signin-email')?.value;
            const password = document.getElementById('signin-password')?.value;

            if (!email || !password) {
                showError('signin-error', 'signin-submit');
                return;
            }
            showSection('intro-section');
        });
    }

    const googleSignup = document.getElementById('google-signup');
    if (googleSignup) {
        googleSignup.addEventListener('click', () => {
            showSection('intro-section');
        });
    }

    // OTP verification
    const verifyOtp = document.getElementById('verify-otp');
    if (verifyOtp) {
        verifyOtp.addEventListener('click', function () {
            const otpInputs = document.querySelectorAll('.otp-input');
            const otpValue = Array.from(otpInputs).map(input => input.value).join('');

            if (otpValue.length === 6) {
                if (otpModal) otpModal.classList.add('hidden');
                showSection('intro-section');
            } else {
                alert("Please enter all 6 digits.");
            }
        });
    }

    // Intro section
    document.addEventListener('click', function (e) {
        if (e.target.classList.contains('next-intro')) {
            currentIntroStep++;
            const currentStep = document.querySelector(`[data-step="${currentIntroStep - 1}"]`);
            const nextStep = document.querySelector(`[data-step="${currentIntroStep}"]`);
            if (currentStep) currentStep.classList.add('hidden');
            if (nextStep) nextStep.classList.remove('hidden');
        }
    });

    const startChat = document.getElementById('start-chat');
    if (startChat) {
        startChat.addEventListener('click', function () {
            showSection('main-interface');
            showMainSection('chat-section');
            if (messageInput) messageInput.focus();
        });
    }

    // Sidebar
    function toggleSidebar() {
        if (sidebar) sidebar.classList.toggle('-translate-x-full');
        if (sidebarOverlay) sidebarOverlay.classList.toggle('hidden');
    }

    const menuToggle = document.getElementById('menu-toggle');
    if (menuToggle) menuToggle.addEventListener('click', toggleSidebar);

    const closeSidebar = document.getElementById('close-sidebar');
    if (closeSidebar) closeSidebar.addEventListener('click', toggleSidebar);

    if (sidebarOverlay) sidebarOverlay.addEventListener('click', toggleSidebar);

    // Navigation
    const navButtons = {
        'nav-chat': 'chat-section',
        'nav-profile': 'profile-section',
        'nav-settings': 'settings-section',
        'nav-support': 'support-section'
    };

    Object.entries(navButtons).forEach(([btnId, sectionId]) => {
        const btn = document.getElementById(btnId);
        if (btn) {
            btn.addEventListener('click', function () {
                Object.values(navButtons).forEach(id => {
                    const section = document.getElementById(id);
                    if (section) section.classList.add('hidden');
                });
                const targetSection = document.getElementById(sectionId);
                if (targetSection) targetSection.classList.remove('hidden');
                toggleSidebar();
            });
        }
    });

    // Logout
    const logoutBtn = document.getElementById('logout-btn');
    if (logoutBtn) {
        logoutBtn.addEventListener('click', function () {
            if (confirm('Are you sure you want to logout?')) {
                chatState = { mode: 'greeting', currentQuestionIndex: 0, collectedData: {} };
                location.reload();
            }
        });
    }

    // ============================================
    // Main Chat Logic
    // ============================================

    if (messageForm) {
        messageForm.addEventListener('submit', async function (e) {
            e.preventDefault();

            if (!isConfigLoaded) {
                addBotMessage("❌ لم يتم تحميل ملفات الإعدادات بعد. يرجى إعادة تحميل الصفحة.");
                return;
            }

            const userMessage = messageInput.value.trim();
            if (!userMessage) return;

            addUserMessage(userMessage);
            messageInput.value = '';

            // Check for farewell
            if (RESPONSES_DATA.farewell_keywords.some(k => userMessage.toLowerCase().includes(k))) {
                await delay(500);
                const farewell = RESPONSES_DATA.farewells[Math.floor(Math.random() * RESPONSES_DATA.farewells.length)];
                addBotMessage(farewell);
                return;
            }

            // State: selecting problem
            if (chatState.mode === 'selecting_problem') {
                const problemNumber = parseInt(userMessage);
                const problems = chatState.solutionsState.problems;

                if (problemNumber >= 1 && problemNumber <= problems.length) {
                    await showProblemSolutions(problemNumber - 1);
                } else {
                    addBotMessage("❌ رقم غير صحيح. اختر رقم من القائمة.");
                }
                return;
            }

            // State: after showing problem solutions
            if (chatState.mode === 'after_problem_solutions') {
                const wantsToContinue = ['نعم', 'آه', 'yes', 'أكيد', 'اه', 'يلا', 'اه'].some(k =>
                    userMessage.toLowerCase().includes(k)
                );

                if (wantsToContinue) {
                    // Return to problems menu
                    await showProblemsMenu();
                } else {
                    // Show farewell message
                    await showFarewellMessages();
                }
                return;
            }

            // State: in interview
            if (chatState.mode === 'in_interview') {
                const currentQuestion = RESPONSES_DATA.interview_questions[chatState.currentQuestionIndex];

                const { reply, storedKey } = getStoredKey(userMessage, currentQuestion);
                chatState.collectedData[currentQuestion.field] = storedKey;

                await delay(500);
                addBotMessage(reply);

                chatState.currentQuestionIndex++;

                if (chatState.currentQuestionIndex < RESPONSES_DATA.interview_questions.length) {
                    const nextQuestion = RESPONSES_DATA.interview_questions[chatState.currentQuestionIndex];
                    await delay(1000);
                    addBotMessage(nextQuestion.question);
                } else {
                    await delay(1000);
                    addBotMessage(RESPONSES_DATA.interview_end);

                    await sendDataToAPI(chatState.collectedData);
                }
            }
            // State: awaiting confirmation
            else if (chatState.mode === 'awaiting_confirmation') {
                const isConfirmed = RESPONSES_DATA.interview_intro.confirmation_keywords.some(k =>
                    userMessage.toLowerCase().includes(k)
                );

                if (isConfirmed) {
                    chatState.mode = 'in_interview';
                    chatState.currentQuestionIndex = 0;
                    await delay(500);
                    addBotMessage(RESPONSES_DATA.interview_questions[0].question);
                } else {
                    chatState.mode = 'greeting';
                    await delay(500);
                    addBotMessage("تمام، براحتك. 😊 لو احتجت حاجة أنا هنا.");
                }
            }
            // State: greeting
            else if (chatState.mode === 'greeting') {
                // Check if greeting
                let isGreeting = false;
                for (const [type, keywords] of Object.entries(RESPONSES_DATA.greetings_keywords)) {
                    if (keywords.some(k => userMessage.toLowerCase().includes(k))) {
                        const greetings = RESPONSES_DATA.greetings[type];
                        const greeting = greetings[Math.floor(Math.random() * greetings.length)];
                        await delay(500);
                        addBotMessage(greeting);
                        isGreeting = true;
                        break;
                    }
                }

                if (!isGreeting) {
                    // Check for negative mood
                    const isNegativeMood = Object.values(RESPONSES_DATA.mood_keywords)
                        .flat()
                        .some(k => userMessage.toLowerCase().includes(k));

                    if (isNegativeMood) {
                        await delay(500);
                        addBotMessage(RESPONSES_DATA.interview_intro.speech);
                        chatState.mode = 'awaiting_confirmation';
                    } else {
                        await delay(500);
                        const unclearMsg = RESPONSES_DATA.unclear_responses[0];
                        addBotMessage(unclearMsg);
                    }
                }
            }
            // State: finished
            else if (chatState.mode === 'finished') {
                await delay(500);
                addBotMessage("المقابلة انتهت بالفعل. لو عايز تبدأ من جديد، قول 'تعبان' أو 'مش كويس'.");
                chatState.mode = 'greeting';
                chatState.collectedData = {};
                chatState.currentQuestionIndex = 0;
                chatState.solutionsState = {
                    problems: [],
                    currentProblemIndex: -1,
                    currentSolutionIndex: 0,
                    currentResourceIndex: 0
                };
            }
        });
    }

    // Initialize app
    showSection('auth-section');
    const signupEmail = document.getElementById('signup-email');
    if (signupEmail) signupEmail.focus();
});