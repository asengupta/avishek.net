<?php 
$root = dirname(dirname(dirname(dirname(dirname(dirname(dirname(__FILE__)))))));
if (file_exists($root.'/wp-load.php')) {
	// WP 2.6
	require_once($root.'/wp-load.php');
} else {
	echo 'alert("Cannot include '.$root.'/wp-load.php!");';
}
Header("content-type: application/x-javascript"); ?>

mySettings = {
    nameSpace:       "html", // Useful to prevent multi-instances CSS conflict
    onShiftEnter:    {keepDefault:false, replaceWith:'<br />\n'},
    onCtrlEnter:     {keepDefault:false, openWith:'\n<p>', closeWith:'</p>\n'},
    onTab:           {keepDefault:false, openWith:'     '},
    markupSet:  [
        {name:'<?php _e('First level heading', 'markitup'); ?>', key:'1', openWith:'<h1(!( class="[![<?php _e('Class', 'markitup'); ?>]!]")!)>', closeWith:'</h1>', placeHolder:'<?php _e('Your title here...', 'markitup'); ?>' },
        {name:'<?php _e('Second level heading', 'markitup'); ?>', key:'2', openWith:'<h2(!( class="[![<?php _e('Class', 'markitup'); ?>]!]")!)>', closeWith:'</h2>', placeHolder:'<?php _e('Your title here...', 'markitup'); ?>' },
        {name:'<?php _e('Heading 3', 'markitup'); ?>', key:'3', openWith:'<h3(!( class="[![<?php _e('Class', 'markitup'); ?>]!]")!)>', closeWith:'</h3>', placeHolder:'<?php _e('Your title here...', 'markitup'); ?>' },
        {name:'<?php _e('Heading 4', 'markitup'); ?>', key:'4', openWith:'<h4(!( class="[![<?php _e('Class', 'markitup'); ?>]!]")!)>', closeWith:'</h4>', placeHolder:'<?php _e('Your title here...', 'markitup'); ?>' },
        {name:'<?php _e('Heading 5', 'markitup'); ?>', key:'5', openWith:'<h5(!( class="[![<?php _e('Class', 'markitup'); ?>]!]")!)>', closeWith:'</h5>', placeHolder:'<?php _e('Your title here...', 'markitup'); ?>' },
        {name:'<?php _e('Heading 6', 'markitup'); ?>', key:'6', openWith:'<h6(!( class="[![<?php _e('Class', 'markitup'); ?>]!]")!)>', closeWith:'</h6>', placeHolder:'<?php _e('Your title here...', 'markitup'); ?>' },
        {name:'<?php _e('Paragraph', 'markitup'); ?>', openWith:'<p(!( class="[![<?php _e('Class', 'markitup'); ?>]!]")!)>', closeWith:'</p>'  },
        {separator:'---------------' },
        {name:'<?php _e('Bold', 'markitup'); ?>', key:'B', openWith:'<strong>', closeWith:'</strong>' },
        {name:'<?php _e('Italic', 'markitup'); ?>', key:'I', openWith:'<em>', closeWith:'</em>'  },
        {name:'<?php _e('Stroke through', 'markitup'); ?>', key:'S', openWith:'<del>', closeWith:'</del>' },
        {separator:'---------------' },
        {name:'<?php _e('Unordered List', 'markitup'); ?>', openWith:'<ul>\n', closeWith:'</ul>\n' },
        {name:'<?php _e('Ordered List', 'markitup'); ?>', openWith:'<ol>\n', closeWith:'</ol>\n' },
        {name:'<?php _e('List item', 'markitup'); ?>', openWith:'<li>', closeWith:'</li>' },
        {separator:'---------------' },
        {name:'<?php _e('Image', 'markitup'); ?>', key:'P', replaceWith:'<img src="[![<?php _e('URL', 'markitup'); ?>:!:http://]!]" alt="[![<?php _e('Alternative text', 'markitup'); ?>]!]" />' },
        {name:'<?php _e('Link', 'markitup'); ?>', key:'L', openWith:'<a href="[![<?php _e('Link', 'markitup'); ?>:!:http://]!]"(!( title="[![<?php _e('Title', 'markitup'); ?>]!]")!)>', closeWith:'</a>', placeHolder:'<?php _e('Your text to link...', 'markitup'); ?>' },
        {separator:'---------------' },
		{name:'<?php _e('Quote block', 'markitup'); ?>', openWith:'<blockquote>\n', closeWith:'</blockquote>\n' },
        {name:'<?php _e('Code block', 'markitup'); ?>', openWith:'<code>\n', closeWith:'</code>\n' },
		{separator:'---------------' },
		{name:'<?php _e('Wordpress More... tag', 'markitup'); ?>', openWith:'\n<!--more-->\n'},
		{separator:'---------------' },
        {name:'<?php _e('Clean', 'markitup'); ?>', replaceWith:function(h) { return h.selection.replace(/<(.*?)>/g, "") } }
    ]
}