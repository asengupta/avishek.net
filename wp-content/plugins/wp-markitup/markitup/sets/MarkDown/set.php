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
	onShiftEnter: {keepDefault:false, openWith:'\n\n'},
	onTab:        {keepDefault:false, replaceWith: function(m) { return miu.multilineStartWith(m, "\t");} },
	markupSet: [
		{name:'<?php _e('First level heading', 'markitup'); ?>', key:'1', placeHolder:'<?php _e('Your title here...', 'markitup'); ?>', closeWith:function(markItUp) { return miu.markdownTitle(markItUp, '=') } },
		{name:'<?php _e('Second level heading', 'markitup'); ?>', key:'2', placeHolder:'<?php _e('Your title here...', 'markitup'); ?>', closeWith:function(markItUp) { return miu.markdownTitle(markItUp, '-') } },
		{name:'<?php _e('Heading 3', 'markitup'); ?>', key:'3', openWith:'### ', placeHolder:'<?php _e('Your title here...', 'markitup'); ?>' },
		{name:'<?php _e('Heading 4', 'markitup'); ?>', key:'4', openWith:'#### ', placeHolder:'<?php _e('Your title here...', 'markitup'); ?>' },
		{name:'<?php _e('Heading 5', 'markitup'); ?>', key:'5', openWith:'##### ', placeHolder:'<?php _e('Your title here...', 'markitup'); ?>' },
		{name:'<?php _e('Heading 6', 'markitup'); ?>', key:'6', openWith:'###### ', placeHolder:'<?php _e('Your title here...', 'markitup'); ?>' },
		{separator:'---------------' },		
		{name:'<?php _e('Bold', 'markitup'); ?>', key:'B', openWith:'**', closeWith:'**'},
		{name:'<?php _e('Italic', 'markitup'); ?>', key:'I', openWith:'_', closeWith:'_'},
		{separator:'---------------' },
		{name:'<?php _e('Bulleted list', 'markitup'); ?>', placeHolder: '<?php _e('Insert your text here...', 'markitup'); ?>', replaceWith: function(m) { return miu.multilineStartWith(m, "- ");} },
		{name:'<?php _e('Numeric list', 'markitup'); ?>', placeHolder: '<?php _e('Insert your text here...', 'markitup'); ?>', replaceWith:function(markItUp) {
			var lines = (markItUp.selection||markItUp.placeHolder).split(/\n/);
			var md_lines = jQuery.map(lines, function(l, i) { return (i+1) + ". " + l; });
			return md_lines.join("\n");
		}},
		{separator:'---------------' },
		{name:'<?php _e('Image', 'markitup'); ?>', key:'P', replaceWith:'![[![<?php _e('Alternative Text', 'markitup'); ?>]!]]([![<?php _e('URL', 'markitup'); ?>:!:http://]!] "[![<?php _e('Title', 'markitup'); ?>]!]")'},
		{name:'<?php _e('Link', 'markitup'); ?>', key:'L', openWith:'[', closeWith:']([![<?php _e('URL', 'markitup'); ?>:!:http://]!] "[![<?php _e('Title', 'markitup'); ?>]!]")', placeHolder: '<?php _e('Insert your text here...', 'markitup'); ?>' },
		{separator:'---------------'},	
		{name:'<?php _e('Quote block', 'markitup'); ?>', replaceWith: function(m) { return miu.multilineStartWith(m, "> ");}, placeHolder: '<?php _e('Insert your text here...', 'markitup'); ?>' },
		{name:'<?php _e('Code block', 'markitup'); ?>', replaceWith: function(m) { 
			if ((m.selection||m.placeHolder).split(/\n/).length > 1) 
				return miu.multilineStartWith(m, "\t");
			else
				return '`' + (m.selection||m.placeHolder) + '`';
			}, placeHolder: '<?php _e('Insert your text here...', 'markitup'); ?>' },
		{separator:'---------------'},
		{name:'<?php _e('Wordpress More... tag', 'markitup'); ?>', openWith:'\n<!--more-->\n'},
	]
}

// mIu nameSpace to avoid conflict.
miu = {
	markdownTitle: function(markItUp, char) {
		heading = '';
		n = jQuery.trim(markItUp.selection||markItUp.placeHolder).length;
		for(i = 0; i < n; i++) {
			heading += char;
		}
		return '\n'+heading;
	},
	multilineStartWith: function(markItUp, char) {
		var lines = (markItUp.selection||markItUp.placeHolder).split(/\n/);
		var md_lines = jQuery.map(lines, function(l, i) { return char + l; });
		return md_lines.join("\n");

	}
}